# TDK dolgozat — mi hiányzik még az 5.2 (in silico) fejezethez

## 1. Ami a poszter-táblázatokból már kész és be van írva
- Rétegi eloszlás (L5 = 62,1%) — `rétegek` munkalap
- PT→GPe / PT→TRN / mindkettő régiónként — `11_`, `10_`, `09_` munkalapok
- Kollateralizáció: 90/435 = 20,7% (GPe-felől), 90/296 = 30,4% (TRN-felől)
- Frontális–motoros vs. szomatoszenzoros szegregáció
- Határon fekvő szómák (border somata) érzékenységi adatai
- Módszertani korlátok — `99_modszertan` munkalap

Ellenőrizve: a `09_L5_BS_GPe_TRN` munkalap sejt-ID listája pontosan a `10_` és
`11_` munkalapok metszete (90 sejt mindkét úton) — a három tábla konzisztens.

## 2. Ami még egy Pályakövető-futtatást igényel

### 2.1 PT-nevező (a legfontosabb)
A poszter-táblázatok nevezője a régió **összes** L5-ös sejtje ("Total Cells"),
nem a PT-populáció. A dolgozat 4.2 fejezete viszont azt írja, hogy az
alappopuláció a leszálló agytörzsbe vetítő sejtek halmaza.
A szoftver ezt már tudja: **Cortical Summary → "Brain stem = 100%" tábla**
(`build_cortical_summary`, `core/analysis.py`). Egy exporttal megkapod, hogy
a *PT-sejtek* hány százaléka vetít a GPe-be / TRN-be / mindkettőbe.
Amíg ez nincs meg, a szövegben mindenhol "az adott régió összes L5-ös
sejtjének %-ában" megfogalmazás áll — ezt kell majd cserélni.

### 2.2 Előtte/utána szám a szigorú kritériumhoz
Ugyanaz a sejtkészlet kétszer lefuttatva:
- min. elágazás = 0 (csak végpont, "laza" kritérium)
- min. elágazás = 1 (szigorú kritérium)
A két GPe-arány különbsége adja a "passage-axon hatás" számszerű mértékét.

### 2.3 Elemszám
Hány SWC-sejt szerepelt összesen az elemzésben (a 4135 csak az L5-ös rész).

## 3. Amit a dolgozat jelenlegi szövegében javítani kell

Az 5.2 fejezet jelenlegi számai nem egyeznek a poszter-táblázatokkal:

| Szövegben szereplő állítás | Tényleges adat |
|---|---|
| "GPe-vetítés ~90–100%-ról ~40%-ra csökkent" | még nincs mérve (lásd 2.2) |
| "frontális/motoros: kb. 45–50% vetít GPe-be és TRN-be is" | 17,0% GPe, 12,1% TRN; **kollaterális: 30,0%** |
| "szomatoszenzoros: 10–15%" | 9,4% GPe, 6,3% TRN; kollaterális: 8,8% |

A "45–50%" vs "10–15%" kontraszt **iránya helyes**, csak a konkrét számok
mások — és a valódi, erősebb állítás a kollateralizációs arány (30,0% vs
8,8%, több mint háromszoros különbség).

## 4. A 4.2 (módszerek) fejezetből hiányzó elemek
A `99_modszertan` munkalap három olyan módszertani döntést dokumentál, ami
még nincs a dolgozatban:
- **`laterality_class` nem használható szűrőként** (az L5 PT-sejtek 62%-át
  tévesen kizárná) — csak leíró statisztikaként
- **`soma_on_region_border`** érzékenységi vizsgálat
- **L6a = negatív kontroll**, nem beválasztási kritérium

---

# In vivo rész (4.1 / 5.1) — a laborjegyzőkönyv alapján

## Ami most már megírható (megírva: `modszerek_invivo.tex`)
- **N = 7 műtét** (GPE16, 17, 19, 20, 22, 23, 24), ebből **2 teljesen kiértékelve**
  (GPE16, GPE24) → ez a dolgozatban szereplő `[N db]` helykitöltő tényleges értéke
- Sztereotaxikus koordináták táblázata (6 állatra megvan)
- Iontoforézis paraméterei: 0,5–1 µA, 2 s be / 2 s ki, 10 perc, 12–19 µm pipetta
- Teljes immunhisztokémiai protokoll: peroxidblokk → 0,2% Triton + 1% NDS →
  nyúl anti-Fluorogold (Chemicon) 1:30 000 → biotinilált anti-nyúl 1:500 →
  ABC → DAB-Ni → krómzselatin → DPX
- 50 µm metszetvastagság, 5 kérgi + 5 talamikus sorozat
- Fluoreszcens sorozat (GPE22–24): PB + Vectashield

## Amit ellenőrizni kell a jegyzőkönyvben
1. **Dátum-hozzárendelés**: 7 állathoz csak 5 műtét- és 5 perfúziódátum van
   megadva. Feltételeztem, hogy GPE22–24 egy napon (08.10) történt (erre utal a
   „12 ul A + 12 ul B a 3 agyra" bejegyzés). → Innen jön a 6–12 napos
   injekció–perfúzió intervallum; ha a hozzárendelés más, ez is változik.
2. **GPE19 koordinátái** hiányoznak (a jegyzőkönyvben „?").
3. **GPE16 iontoforézis árama**: a megjegyzés szerint „valószínűleg rosszul volt
   beállítva a beadó, és nem 0,5 µA áram volt, hanem 50" — ezt tisztázni kell,
   mert a táblázatban jelenleg 0,5 µA szerepel.
4. **ML előjelek**: GPE23 és GPE24 esetében negatív (−2,4) — feltehetően bal
   félteke. Érdemes egységesíteni, vagy a táblázatban jelölni a féltekét.
5. A PDF 2–13. oldala üres (a táblázat exportja csak az 1. oldalra fért rá) —
   ha van további tartalom, azt külön kell kinyerni.

## 5.1 (in vivo eredmények) — mi kell még
- A GPE16 és GPE24 slide scanner / konfokális képei (ábraként)
- Az injekciós helyek berajzolása egy referencia-atlasz metszeteire
- A megjelölt kérgi régiók listája és rétegi eloszlása a két kiértékelt állatból
- Összevetés az Abecassis és mtsai. (2020) SSp/MOp-dominálta mintázatával
