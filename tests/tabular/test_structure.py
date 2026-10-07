"""Table structure parsing protects cell spans and grid dimensions."""

import unittest

from rag1.extractions.tabular.structure import parse_structure


class StructureTests(unittest.TestCase):
    def test_parses_rowspan_and_colspan_without_losing_columns(self) -> None:
        grid = parse_structure([
            "<html>", "<body>", "<table>",
            "<tr>", '<td colspan="2">', "</td>", "<td></td>", "</tr>",
            "<tr>", '<td rowspan="2">', "</td>", "<td></td>", "<td></td>", "</tr>",
            "<tr>", "<td></td>", "<td></td>", "</tr>",
            "</table>", "</body>", "</html>",
        ])

        self.assertEqual(grid.column_count, 3)
        self.assertEqual(len(grid.rows), 3)
        self.assertEqual(grid.rows[0][0].colspan, 2)
        self.assertEqual(grid.rows[1][0].rowspan, 2)
        self.assertEqual([cell.column for cell in grid.rows[2]], [1, 2])

    def test_rejects_inconsistent_width(self) -> None:
        with self.assertRaises(ValueError):
            parse_structure(["<table><tr><td></td></tr><tr><td></td><td></td></tr></table>"])


if __name__ == "__main__":
    unittest.main()
