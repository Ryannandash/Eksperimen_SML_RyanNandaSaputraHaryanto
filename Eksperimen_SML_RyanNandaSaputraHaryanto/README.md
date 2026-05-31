# Eksperimen_SML_Nama-siswa

Repositori ini berisi eksperimen dan otomatisasi preprocessing dataset **Sports Car** untuk submission kelas Machine Learning.

---

## Struktur Folder

```
Eksperimen_SML_Nama-siswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       ← GitHub Actions (Advance)
├── dataset_raw.csv                 ← Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Nama-siswa.ipynb ← Notebook eksperimen (Basic)
│   ├── automate_Nama-siswa.py      ← Script otomatisasi (Skilled)
│   └── dataset_preprocessing/
│       └── dataset_preprocessing.csv ← Output hasil preprocessing
└── README.md
```

---

## Dataset

**Sports Car Price Dataset**
- Jumlah baris raw : 1007
- Jumlah kolom     : 8
- Target           : `Price (in USD)` — Regresi harga mobil

| Kolom | Deskripsi |
|-------|-----------|
| Car Make | Merek mobil |
| Car Model | Model mobil |
| Year | Tahun produksi |
| Engine Size (L) | Ukuran mesin (liter) |
| Horsepower | Tenaga kuda |
| Torque (lb-ft) | Torsi |
| 0-60 MPH Time (seconds) | Waktu akselerasi 0-60 mph |
| Price (in USD) | Harga mobil (target) |

---

## Tahapan Preprocessing

1. **Konversi Tipe Data** — kolom numerik dari string → float, Price hapus koma
2. **Hapus Duplikat** — 292 baris duplikat dihapus
3. **Handle Missing Values** — imputation dengan median
4. **Remove Outlier** — metode IQR (1.5×IQR)
5. **Label Encoding** — Car Make & Car Model → integer
6. **StandardScaler** — normalisasi seluruh fitur

**Hasil:** 607 baris bersih, siap latih.

---

## Cara Menjalankan Preprocessing Manual

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Jalankan dari folder preprocessing/
python automate_Nama-siswa.py --input ../dataset_raw.csv --output dataset_preprocessing/dataset_preprocessing.csv
```

---

## GitHub Actions (Advance)

Workflow otomatis terpantik setiap kali:
- Ada push ke `main` yang mengubah `dataset_raw.csv` atau `automate_Nama-siswa.py`
- Trigger manual via GitHub → Actions → Run workflow

Workflow akan otomatis menjalankan preprocessing dan commit hasilnya ke repo.
