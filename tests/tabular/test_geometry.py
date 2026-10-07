"""OCR geometry can recover a column missed by structure recognition."""

import unittest

from rag1.extractions.tabular.geometry import TextBox, build_geometry_grid, infer_column_anchors


class GeometryTests(unittest.TestCase):
    def test_two_row_table_keeps_ocr_columns(self) -> None:
        boxes = [
            TextBox(f"{row}-{col}", str(col), (x, 30 + row * 40, x + 20, 48 + row * 40))
            for row in range(2) for col, x in enumerate((100, 280))
        ]
        self.assertEqual(len(infer_column_anchors(boxes, (90, 20, 360, 120))), 2)

    def test_five_stable_columns_override_a_four_column_structure(self) -> None:
        starts = (180, 260, 850, 1040, 1350)
        boxes = [
            TextBox(
                ref=f"line-{row}-{column}",
                text=f"{row}:{column}",
                bbox=(x, 100 + row * 48, x + 24, 124 + row * 48),
            )
            for row in range(5)
            for column, x in enumerate(starts)
        ]

        anchors = infer_column_anchors(boxes, (170, 90, 1510, 350))
        grid, unassigned = build_geometry_grid(boxes, anchors)

        self.assertEqual(len(anchors), 5)
        self.assertEqual(grid.column_count, 5)
        self.assertEqual(len(grid.rows), 5)
        self.assertEqual(grid.rows[0][0].text, "0:0")
        self.assertEqual(grid.rows[4][4].text, "4:4")
        self.assertEqual(unassigned, [])


if __name__ == "__main__":
    unittest.main()
