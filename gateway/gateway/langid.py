"""Language identification for the Multilingual Unit Tests evaluator.

Ports the scoring half of the code-switching benchmark: given a reply, which of
English / Chinese / Malay / Tamil is it in?

**fastText parity when it's available.** The benchmark scores with fastText
`lid.176`, and its published numbers depend on that model's exact behaviour —
including its quirks. Point `EXPERIMENT_FASTTEXT_MODEL` at a `lid.176.bin` (with
the `fasttext` wheel installed) and this module uses it, mapping labels exactly
as the benchmark does. Otherwise it falls back to the built-in detector below,
and every result records which path ran (`detector: "fasttext" | "builtin"`) so
two runs are never silently incomparable.

**The built-in detector** needs no model download. Chinese and Tamil are settled
by script alone (they have their own Unicode blocks, so this is exact). Malay vs
English is the only real decision, and it's made on **function words** — `yang`,
`dan`, `untuk` vs `the`, `and`, `for` — which are the reliable signal in
code-switched Manglish text where content words borrow freely from English.

⚠ **`__label__id` maps to `malay`, exactly as the benchmark does** — fastText
cannot reliably separate Malaysian Malay from Indonesian. That inflates Malay
accuracy, which is why `indonesian_leak()` exists: a curated lexicon of words
that are distinctly Indonesian and NOT valid Malay, matched whole-token so
`uang` doesn't fire inside `ruang`. The evaluator reports the corrected number
alongside the raw one rather than quietly picking either.
"""
from __future__ import annotations

import os
import re
from typing import Optional

LANGUAGES = ("english", "chinese", "malay", "tamil")

# fastText label → our language name. `id` → malay is deliberate (see docstring).
_LABEL_MAP = {
    "__label__en": "english",
    "__label__zh": "chinese",
    "__label__ta": "tamil",
    "__label__ms": "malay",
    "__label__id": "malay",
}

_FASTTEXT_ENV = "EXPERIMENT_FASTTEXT_MODEL"
_ft_model = None
_ft_tried = False


def _fasttext():
    """Load lid.176 once, if configured and importable. None = use the fallback."""
    global _ft_model, _ft_tried
    if _ft_tried:
        return _ft_model
    _ft_tried = True
    path = (os.environ.get(_FASTTEXT_ENV, "") or "").strip()
    if not path or not os.path.exists(path):
        return None
    try:
        import fasttext  # noqa: PLC0415 - optional dependency
        _ft_model = fasttext.load_model(path)
    except Exception:
        _ft_model = None
    return _ft_model


# --------------------------------------------------------------------------- #
# Script detection — exact for Chinese and Tamil
# --------------------------------------------------------------------------- #

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_TAMIL = re.compile(r"[\u0b80-\u0bff]")
_LATIN = re.compile(r"[A-Za-z]")
# Scripts we can name but don't score — reported as-is rather than guessed at.
_OTHER_SCRIPTS = (
    ("japanese", re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")),
    ("korean", re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")),
    ("arabic", re.compile(r"[\u0600-\u06ff]")),
    ("thai", re.compile(r"[\u0e00-\u0e7f]")),
    ("devanagari", re.compile(r"[\u0900-\u097f]")),
    ("cyrillic", re.compile(r"[\u0400-\u04ff]")),
)

# Function words. Content words borrow freely across Manglish, so these carry
# the signal; a word in BOTH lists would be noise and is excluded.
_MALAY_MARKERS = frozenset("""
yang dan untuk dengan tidak adalah akan boleh saya anda kami kita ini itu pada dari
dalam atau juga sila sudah telah tetapi kerana bagi oleh ke di ada apa siapa bila
bagaimana mengapa kenapa berapa nak hendak jangan belum masih lagi sahaja hanya
semua setiap beberapa banyak sedikit lebih paling sangat amat terima kasih maaf
tolong bantu bantuan nombor akaun perkhidmatan maklumat pelanggan pembayaran bayar
bulan tarikh masa sekarang esok semalam hari minggu tahun rumah kereta orang tuan
puan encik cik anda kepada daripada tersebut berikut seperti supaya agar walaupun
kalau jika sekiranya iaitu ialah bukan belum pernah sila semak pastikan
""".split())

_ENGLISH_MARKERS = frozenset("""
the and for with not is are was were will can could should would you your yours we
our this that these those from into about over under there here what who when where
why how which have has had been being does did doing please thank thanks sorry help
account number service customer payment pay month date time now today tomorrow
yesterday day week year but because so if then than they them their his her its
been also just only all any some many more most very much need want make made take
take get got give given send sent check checked
""".split())

_WORD = re.compile(r"[a-z']+")

# Distinctly Indonesian words that are NOT valid Malaysian Malay, ported from the
# benchmark's curated lexicon. Whole-token matching only — see the docstring.
_INDONESIAN_ONLY = frozenset({
    "agunan", "aja", "aksesibilitas", "aktivasi", "aktivitas", "akuntabilitas", "angkot",
    "angsuran", "anjay", "anjir", "antre", "antrean", "antri", "apaan", "apotek", "apotik",
    "asik", "asin", "asuransi", "bagian", "bahwa", "bakso", "bandara", "banget", "bantuin",
    "baper", "barusan", "baterai", "bayarin", "beasiswa", "becak", "beda", "begal",
    "belanjaan", "belom", "bener", "benerin", "bensin", "bentar", "beraktivitas", "berbeda",
    "beresin", "berhasil", "berkualitas", "berlangganan", "besok", "betah", "bete", "biaya",
    "bioskop", "bisnis", "blokir", "bocah", "bohlam", "bokap", "bokek", "boong", "bpjs",
    "bule", "cakep", "cape", "capek", "cave", "celana", "cemilan", "cepet", "cewek",
    "cicilan", "coba", "colokan", "cowok", "cuek", "cuman", "curhat", "dapet", "dasi",
    "dateng", "deket", "denger", "dengerin", "dikasih", "dikit", "dikonfirmasi", "diperbarui",
    "diprioritaskan", "diskon", "diversitas", "doang", "doi", "dokter", "dosen",
    "efektivitas", "eksklusivitas", "ekspor", "elo", "elu", "emang", "emangnya", "engga",
    "enggak", "entar", "fasilitas", "fitur", "fleksibilitas", "formalitas", "formulir",
    "fotokopi", "fungsionalitas", "gabut", "gajian", "ganteng", "gausah", "gebetan", "gede",
    "gemes", "gengsi", "gercep", "gimana", "gini", "gitu", "goblok", "gokil", "gorden",
    "gorengan", "gpp", "gratis", "gue", "gurih", "halte", "handuk", "hewan", "identitas",
    "impor", "imut", "inget", "integritas", "intensitas", "intinya", "istri", "iuran", "iya",
    "jadwal", "jagoan", "jajan", "jelek", "jilbab", "jomblo", "jorok", "jumat", "jutek",
    "kabupaten", "kacamata", "kadaluarsa", "kadaluwarsa", "kaga", "kagak", "kaget", "kakek",
    "kalian", "kalo", "kamar", "kamis", "kampanye", "kantong", "kantor", "kaos",
    "kapabilitas", "kapasitas", "karcis", "karena", "kartu", "karyawan", "kasir", "kasur",
    "kasus", "kaus", "kayak", "kayaknya", "kebijakan", "kebutuhan", "kecamatan", "kecepatan",
    "kedaluwarsa", "keluhan", "kelupaan", "kelurahan", "kemarin", "kembalian", "kendala",
    "kendaraan", "kepengen", "keponakan", "keran", "keren", "kerudung", "kesel", "ketemu",
    "ketemuan", "ketentuan", "keterlambatan", "ketidaknyamanan", "keuangan", "khawatir",
    "klakson", "kode", "komoditas", "kompatibilitas", "kompleksitas", "kompor", "komunitas",
    "koneksi", "konfirmasi", "konsumen", "kontinuitas", "kosan", "kost", "kreativitas",
    "kredibilitas", "kriminalitas", "kualitas", "kuantitas", "kuatir", "kuitansi", "kulkas",
    "kursi", "kuy", "kwitansi", "lagian", "laper", "legalitas", "legitimitas", "lemari",
    "lembur", "lemot", "listrik", "losmen", "loyalitas", "macet", "mager", "makanya",
    "makasi", "makasih", "maling", "mampir", "mantul", "maskapai", "materai", "mayoritas",
    "mbak", "melunasi", "memblokir", "membutuhkan", "memperbarui", "memprioritaskan",
    "mencoba", "mending", "mendingan", "mengabari", "mengajukan", "menginformasikan",
    "mengonfirmasi", "menindaklanjuti", "menonaktifkan", "mentalitas", "menunda", "merubah",
    "mesen", "meterai", "metoda", "metode", "mie", "mikir", "minoritas", "mobil", "mobilitas",
    "moralitas", "mortalitas", "mudik", "mulu", "mumpung", "nabung", "nanya", "nanyain",
    "napas", "naruh", "nasabah", "nasehat", "nasionalitas", "nelpon", "ngabarin", "ngajak",
    "ngajakin", "ngambek", "ngambil", "nganter", "nganterin", "ngapain", "ngasih", "ngebut",
    "ngecas", "ngecek", "ngeliat", "ngepel", "ngerti", "ngga", "nggak", "nginep", "nginget",
    "ngirim", "ngirimin", "ngobrol", "ngomel", "ngomong", "ngomongin", "nomer", "nomor",
    "nonaktif", "nongkrong", "ntar", "nunggu", "nungguin", "nyampe", "nyantai", "nyari",
    "nyetir", "nyetor", "nyokap", "nyolong", "nyuci", "nyuruh", "obat", "obesitas",
    "objektivitas", "obral", "obrolan", "odol", "ojek", "omong", "ongkir", "ongkos",
    "operasional", "ortu", "otomatis", "otoritas", "pake", "panci", "parkir", "pasien",
    "pelunasan", "pembaruan", "pemesanan", "pengaduan", "pengajuan", "pengecekan",
    "pengembalian", "pengen", "pengin", "penjadwalan", "perawat", "perbarui", "perbedaan",
    "permisi", "persentase", "personalitas", "pertanggungan", "pesen", "pikir", "pilek",
    "pingin", "pinter", "plafon", "pokoknya", "ponsel", "popularitas", "populer", "praktek",
    "praktik", "preman", "pria", "pribumi", "prioritas", "probabilitas", "produktivitas",
    "provinsi", "pulpen", "pulsa", "puskesmas", "rame", "rapor", "rasionalitas", "realitas",
    "receh", "registrasi", "rekening", "relativitas", "reliabilitas", "repot", "resep",
    "resmi", "ribet", "rincian", "rok", "rusak", "rute", "rutinitas", "saklar", "santunan",
    "sebagian", "sebel", "seksualitas", "sendok", "seneng", "senin", "sensitivitas", "sepatu",
    "sepeda", "seprai", "setir", "setor", "setoran", "sinyal", "solidaritas", "sopir",
    "sortir", "spiritualitas", "sprei", "stabilitas", "standar", "stasiun", "stempel",
    "stopkontak", "struk", "subjektivitas", "subtitle", "sukses", "survei", "suster",
    "tabungan", "tagihan", "tante", "tautan", "teknis", "telat", "telepon", "televisi",
    "temen", "terdaftar", "terhubung", "terjadwal", "terkirim", "tertanggung", "tertunda",
    "tivi", "toko", "tombol", "totalitas", "trus", "tuh", "tunda", "uang", "udah", "udh",
    "ulangan", "unduh", "unduhan", "unggah", "unggahan", "universitas", "utang", "validitas",
    "vitalitas", "wastafel", "wortel", "yaudah", "yok", "yuk",
})


def _script_language(text: str) -> Optional[str]:
    """Language settled by script alone, or None when it's Latin/ambiguous."""
    cjk, tamil, latin = len(_CJK.findall(text)), len(_TAMIL.findall(text)), len(_LATIN.findall(text))
    # A Manglish reply can carry a stray CJK glyph; require the script to
    # actually dominate before calling it.
    if tamil and tamil >= latin:
        return "tamil"
    if cjk and cjk * 2 >= latin:
        return "chinese"
    for name, pattern in _OTHER_SCRIPTS:
        hits = len(pattern.findall(text))
        if hits and hits >= latin:
            return name
    return None


def detect_language(text: str) -> tuple[str, float]:
    """(language, confidence). "unknown" for empty/undecidable text.

    Confidence is fastText's own score when that path is active, else the share
    of marker words that pointed at the winner.
    """
    text = (text or "").strip()
    if not text:
        return "unknown", 0.0

    model = _fasttext()
    if model is not None:
        try:
            labels, scores = model.predict(text.replace("\n", " "))
            if labels:
                return _LABEL_MAP.get(labels[0], "unknown"), float(scores[0])
        except Exception:
            pass  # fall through to the built-in detector

    by_script = _script_language(text)
    if by_script:
        return by_script, 1.0

    words = _WORD.findall(text.lower())
    if not words:
        return "unknown", 0.0
    ms = sum(1 for w in words if w in _MALAY_MARKERS)
    en = sum(1 for w in words if w in _ENGLISH_MARKERS)
    if ms == 0 and en == 0:
        # Latin script with no function words at all (a bare code block, an id).
        return "unknown", 0.0
    if ms == en:
        # A genuine tie leans English: Manglish borrows English content words, so
        # equal marker counts usually means an English reply with Malay borrowings.
        return "english", 0.5
    winner = "malay" if ms > en else "english"
    return winner, round(max(ms, en) / (ms + en), 3)


def indonesian_leak(text: str) -> list[str]:
    """Distinctly-Indonesian tokens in `text`, whole-token and case-insensitive.

    Non-empty means a reply scored as `malay` is really Indonesian — the leak the
    benchmark's corrected Malay column removes.
    """
    words = set(_WORD.findall((text or "").lower()))
    return sorted(words & _INDONESIAN_ONLY)


def detector_name() -> str:
    """Which path detect_language() will take — recorded on every result so two
    runs scored differently are never silently compared."""
    return "fasttext" if _fasttext() is not None else "builtin"
