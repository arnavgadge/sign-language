/*
  lcd_display.ino
  ---------------
  AE-CP  —  Arduino receiver side.

  Python sends the growing word over Serial each time a new letter
  is accepted (e.g. "H", then "HI", then "HIT").

  LCD shows:
    Line 1 →  last accepted letter  (large, centred)
    Line 2 →  full word so far

  Hardware
  --------
  Arduino UNO + 16×2 LCD with I2C backpack (PCF8574)

  I2C wiring
  ----------
  VCC  → 5V
  GND  → GND
  SDA  → A4
  SCL  → A5

  Library
  -------
  LiquidCrystal_I2C  (by Frank de Brabander — install via Library Manager)

  I2C address
  -----------
  Most modules use 0x27.  If LCD stays blank, run I2C Scanner sketch
  to find the real address and replace 0x27 below (common alt: 0x3F).
*/

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

String incoming = "";
bool   newWord  = false;


// ════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);

  lcd.init();
  lcd.backlight();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("  AE-CP Ready");
  lcd.setCursor(0, 1);
  lcd.print(" Waiting...");

  Serial.println("[ARDUINO] Ready.");
}


// ════════════════════════════════════════════════════════
//  LOOP
// ════════════════════════════════════════════════════════
void loop() {

  // Read serial bytes until newline
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      incoming.trim();
      if (incoming.length() > 0) newWord = true;
    } else {
      incoming += c;
    }
  }

  if (newWord) {
    displayUpdate(incoming);
    Serial.print("[ARDUINO] Word: ");
    Serial.println(incoming);
    incoming = "";
    newWord  = false;
  }
}


// ════════════════════════════════════════════════════════
//  DISPLAY
// ════════════════════════════════════════════════════════
void displayUpdate(String word) {
  int len = word.length();
  if (len == 0) return;

  // Last accepted letter
  String lastLetter = String(word[len - 1]);

  // Line 1: last letter centred  e.g. "       H       "
  lcd.setCursor(0, 0);
  lcd.print("                ");          // clear line 1
  int col = (16 - 1) / 2;               // centre of 16 cols
  lcd.setCursor(col, 0);
  lcd.print(lastLetter);

  // Line 2: full word (truncate at 16 chars)
  lcd.setCursor(0, 1);
  lcd.print("                ");          // clear line 2
  lcd.setCursor(0, 1);
  if (len <= 16) {
    lcd.print(word);
  } else {
    // Show the last 16 chars so the newest letter is always visible
    lcd.print(word.substring(len - 16));
  }
}
