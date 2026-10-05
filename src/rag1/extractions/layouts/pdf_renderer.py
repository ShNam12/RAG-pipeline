"""PDF page rendering and layout overlay artifacts for debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from rag1.extractions.layouts.contracts import Region, RegionKind, VisualLocation
from rag1.extractions.layouts.paddle import (
    ModelFactory,
    PaddleLayoutAdapter,
    RenderedPage,
)
from rag1.extractions.layouts.writer import document_output_path


DEFAULT_RENDER_DPI = 200
DEFAULT_PAGE_IMAGE_DIR = Path("pages")
BBOX_IMAGE_DIR = Path("bbox")
_REGION_COLORS = {
    RegionKind.TEXT: (255, 196, 0),
    RegionKind.TABLE: (255, 48, 48),
    RegionKind.PICTURE: (0, 180, 255),
    RegionKind.OTHER: (255, 0, 255),
}


class PDFPageRenderer:
    """Render PDF pages to deterministic PNGs and optional bbox overlays."""

    def __init__(
        self,
        *,
        dpi: int = DEFAULT_RENDER_DPI,
        output_dir: str | Path = Path("data/extraction/pdf_debug"),
        image_dir: str | Path = DEFAULT_PAGE_IMAGE_DIR,
    ) -> None:
        if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
            raise ValueError("dpi must be a positive integer")
        image_subdir = Path(image_dir)
        if (
            image_subdir.is_absolute()
            or not image_subdir.parts
            or any(part in {".", ".."} for part in image_subdir.parts)
        ):
            raise ValueError("image_dir must be a relative subdirectory without '..'")
        if image_subdir.parts[0].casefold() == BBOX_IMAGE_DIR.name:
            raise ValueError("image_dir must not overlap the reserved 'bbox' directory")
        self.dpi = dpi
        self.output_dir = Path(output_dir)
        self.image_dir = image_subdir
        self._last_document_dir: Path | None = None

    def render(
        self,
        source: str | Path,
        *,
        document_output_dir: str | Path | None = None,
    ) -> list[RenderedPage]:
        """Render every page and save a manifest with dimensions and rotation."""
        pdf_path = Path(source)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF page renderer requires a .pdf source: {pdf_path}")

        try:
            import pypdfium2 as pdfium
        except ImportError as error:
            raise RuntimeError("PDF page rendering requires pypdfium2") from error

        document_dir = (
            Path(document_output_dir)
            if document_output_dir is not None
            else self._document_output_dir(pdf_path)
        )
        document_dir.mkdir(parents=True, exist_ok=True)
        image_dir = document_dir / self.image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        self._last_document_dir = document_dir
        rendered_pages: list[RenderedPage] = []
        page_metadata: list[dict[str, object]] = []

        pdf_document = pdfium.PdfDocument(str(pdf_path))
        try:
            for page_index in range(len(pdf_document)):
                page_number = page_index + 1
                pdf_page = pdf_document[page_index]
                try:
                    rotation_degrees = pdf_page.get_rotation()
                    bitmap = pdf_page.render(scale=self.dpi / 72)
                    try:
                        image = bitmap.to_pil().convert("RGB")
                        image_name = f"page-{page_number:04d}.png"
                        image_path = image_dir / image_name
                        image.save(image_path, format="PNG")
                        rendered_pages.append(
                            RenderedPage(
                                page_number=page_number,
                                image=str(image_path),
                                page_width=image.width,
                                page_height=image.height,
                                rotation_degrees=rotation_degrees,
                            )
                        )
                        page_metadata.append(
                            {
                                "page_number": page_number,
                                "rotation_degrees": rotation_degrees,
                                "pixel_width": image.width,
                                "pixel_height": image.height,
                                "image_name": image_path.relative_to(document_dir).as_posix(),
                            }
                        )
                    finally:
                        bitmap.close()
                finally:
                    pdf_page.close()
        finally:
            pdf_document.close()

        self._write_render_manifest(pdf_path, page_metadata, document_dir)
        return rendered_pages

    def export_bbox_overlays(
        self,
        pages: Sequence[RenderedPage],
        regions: Sequence[Region],
        *,
        document_output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Save page copies with matching visual regions outlined."""
        if not pages:
            return []
        document_dir = (
            Path(document_output_dir)
            if document_output_dir is not None
            else self._last_document_dir
        )
        if document_dir is None:
            raise ValueError("render pages before exporting bbox overlays")

        bbox_dir = document_dir / BBOX_IMAGE_DIR
        bbox_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        manifest_pages: list[dict[str, object]] = []

        for page in pages:
            if not isinstance(page.image, (str, Path)):
                raise ValueError(
                    f"Rendered page {page.page_number} does not reference a saved image"
                )
            image_path = Path(page.image)

            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            draw = ImageDraw.Draw(image)
            matching_regions = [
                region
                for region in regions
                if isinstance(region.location, VisualLocation)
                and region.location.page_number == page.page_number
            ]
            for region in matching_regions:
                location = region.location
                assert isinstance(location, VisualLocation)
                x_scale = image.width / location.page_width
                y_scale = image.height / location.page_height
                x0, y0, x1, y1 = location.bbox
                box = (
                    round(x0 * x_scale),
                    round(y0 * y_scale),
                    round(x1 * x_scale),
                    round(y1 * y_scale),
                )
                color = _REGION_COLORS[region.kind]
                draw.rectangle(box, outline=color, width=max(2, self.dpi // 80))
                label = f"{region.kind.value}:{region.id}"
                font = ImageFont.load_default()
                label_box = draw.textbbox((box[0], box[1]), label, font=font)
                draw.rectangle(label_box, fill=color)
                draw.text((box[0], box[1]), label, fill=(0, 0, 0), font=font)

            image_name = image_path.name
            bbox_image_name = f"{Path(image_name).stem}_bbox.png"
            bbox_image_path = bbox_dir / bbox_image_name
            image.save(bbox_image_path, format="PNG")
            output_paths.append(bbox_image_path)
            manifest_pages.append(
                {
                    "page_number": page.page_number,
                    "image_name": image_path.relative_to(document_dir).as_posix(),
                    "bbox_image_name": bbox_image_path.relative_to(document_dir).as_posix(),
                    "region_count": len(matching_regions),
                    "region_ids": [region.id for region in matching_regions],
                }
            )

        if pages:
            self._update_render_manifest(document_dir, manifest_pages)
        return output_paths

    def _document_output_dir(self, source: Path) -> Path:
        return document_output_path(self.output_dir, source)

    def _write_render_manifest(
        self,
        source: Path,
        pages: Sequence[dict[str, object]],
        document_dir: Path,
    ) -> None:
        manifest = {
            "schema_version": "1.0",
            "source": str(source.resolve()),
            "dpi": self.dpi,
            "pages": [
                {**page, "bbox_image_name": None}
                for page in pages
            ],
        }
        (document_dir / "render-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _update_render_manifest(
        document_dir: Path,
        overlay_pages: Sequence[dict[str, object]],
    ) -> None:
        manifest_path = document_dir / "render-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        overlays_by_page = {
            int(page["page_number"]): page for page in overlay_pages
        }
        for page in manifest["pages"]:
            overlay = overlays_by_page.get(page["page_number"])
            if overlay is not None:
                page["bbox_image_name"] = overlay["bbox_image_name"]
                page["region_count"] = overlay["region_count"]
                page["region_ids"] = overlay["region_ids"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def render_pdf_layout_debug(
    source: str | Path,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    output_dir: str | Path = Path("data/extraction/pdf_debug"),
    image_dir: str | Path = DEFAULT_PAGE_IMAGE_DIR,
    model_factory: ModelFactory | None = None,
) -> Path:
    """Render, detect, and save page images with region boxes for inspection."""
    pdf_path = Path(source)
    renderer = PDFPageRenderer(dpi=dpi, output_dir=output_dir, image_dir=image_dir)
    pages = renderer.render(pdf_path)
    document = PaddleLayoutAdapter(model_factory=model_factory).extract_document(
        source=str(pdf_path),
        source_format="pdf",
        pages=pages,
    )
    renderer.export_bbox_overlays(pages, document.regions)

    document_dir = renderer._document_output_dir(pdf_path)
    document_path = document_dir / "layout.json"
    document_path.write_text(
        document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return document_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a PDF, detect layout regions, and save bbox debug images."
    )
    parser.add_argument("source", type=Path, help="PDF to render and inspect")
    parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/extraction/pdf_debug"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_PAGE_IMAGE_DIR,
        help="relative subdirectory for unannotated page images (default: pages)",
    )
    arguments = parser.parse_args()
    document_path = render_pdf_layout_debug(
        arguments.source,
        dpi=arguments.dpi,
        output_dir=arguments.output_dir,
        image_dir=arguments.image_dir,
    )
    print(f"Saved PDF layout debug output to {document_path.parent}")


if __name__ == "__main__":
    main()
