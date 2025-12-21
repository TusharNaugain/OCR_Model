
import unittest
from ocr_utils import clean_ocr_noise, fix_character_confusion

class TestOCRCleanup(unittest.TestCase):
    def test_stock_holding_fix(self):
        raw = "4. The of this Sums Vb a Mobile App of Stock Hotding."
        cleaned = clean_ocr_noise(raw)
        self.assertIn("Stock Holding", cleaned)

    def test_date_fix(self):
        raw = "Date 2026 G4 18.PM 81-Dec-2125 Bk" # 81-Dec is physically impossible, likely 31-Dec
        cleaned = clean_ocr_noise(raw)
        self.assertIn("31-Dec-2125", cleaned)

    def test_noise_removal(self):
        raw = """
fix it Acetate Meta AiR j
no
=>. is
ORES
NS
7,
Account Reterence
"""
        cleaned = clean_ocr_noise(raw)
        # Should remove the garbage lines
        self.assertNotIn("=>.", cleaned)
        self.assertNotIn("7,", cleaned)
        self.assertIn("Account Reference", cleaned)

if __name__ == '__main__':
    unittest.main()
