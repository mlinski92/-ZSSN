"""
legal_chunker.py
================
Inteligentny podział (chunking) polskich aktów prawnych.

Hierarchia struktury aktu prawnego:
    Ustawa / Rozporządzenie / Uchwała
    └── Część / Dział / Rozdział / Oddział
        └── Artykuł (Art.)  ← główna jednostka redakcyjna
            └── Paragraf (§)
                └── Ustęp (ust.)
                    └── Punkt (pkt / 1. / a))
                        └── Litera (lit.)

Strategia:
    1. Parsowanie struktury → wyodrębnienie metadanych każdego fragmentu
    2. Podział na chunk-i WZDŁUŻ granic strukturalnych (nie w środku artykułu)
    3. Każdy chunk niesie pełny kontekst (tytuł aktu, rozdział, artykuł)
    4. Overlap = powtórzenie ostatniego artykułu/paragrafu (nie surowych znaków)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Wzorce regex dla polskich aktów prawnych
# ---------------------------------------------------------------------------

# Nagłówki strukturalne (Części, Działy, Rozdziały, Oddziały)
_RE_SECTION = re.compile(
    r"^(?P<type>CZ[ĘE][ŚS][ĆC]|DZIA[ŁL]|ROZDZIA[ŁL]|ODZIA[ŁL])\s+"
    r"(?P<num>[IVXLCDM\d]+\.?)\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Artykuł: "Art. 1." lub "Art. 1a." lub "Artykuł 1"
_RE_ARTICLE = re.compile(
    r"^(?P<kw>Art(?:ykuł)?\.?)\s+(?P<num>\d+\w*)\s*\.",
    re.IGNORECASE | re.MULTILINE,
)

# Paragraf: "§ 1." lub "§1."
_RE_PARA = re.compile(
    r"^§\s*(?P<num>\d+\w*)\s*\.",
    re.MULTILINE,
)

# Ustęp: "1." na początku linii (liczba + kropka, nie będąca artykułem)
_RE_USTEP = re.compile(
    r"^(?P<num>\d+)\.\s",
    re.MULTILINE,
)

# Tytuł aktu (pierwsze zdanie zwykle po nagłówku USTAWA / ROZPORZĄDZENIE)
_RE_TITLE = re.compile(
    r"^(USTAWA|ROZPORZĄDZENIE|UCHWAŁA|ZARZĄDZENIE|DYREKTYWA|TRAKTAT)"
    r"(?:\s+\S+){0,20}",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Struktury danych
# ---------------------------------------------------------------------------

@dataclass
class LegalUnit:
    """Pojedyncza jednostka redakcyjna aktu (artykuł, paragraf, itp.)."""
    unit_type: str          # "article" | "paragraph" | "section" | "preamble"
    number: str             # "1", "1a", "§ 5", itp.
    text: str               # pełna treść jednostki
    # Kontekst nadrzędny
    act_title: str = ""
    section: str = ""       # np. "Rozdział 2 – Zasady ogólne"
    # Pozycja w dokumencie
    char_start: int = 0
    char_end: int = 0
    # Dodatkowe metadane
    source_file: str = ""
    page_hint: int = 0      # przybliżona strona (jeśli znana)

    @property
    def reference(self) -> str:
        """Czytelne odwołanie, np. 'Rozdział 2, Art. 15'."""
        parts = []
        if self.section:
            parts.append(self.section)
        if self.unit_type == "article":
            parts.append(f"Art. {self.number}")
        elif self.unit_type == "paragraph":
            parts.append(f"§ {self.number}")
        return ", ".join(parts) if parts else self.number

    @property
    def full_context_text(self) -> str:
        """Tekst wzbogacony o metadane – gotowy do embeddingu."""
        header = f"[{self.act_title}] {self.reference}\n" if self.act_title else f"{self.reference}\n"
        return header + self.text.strip()


@dataclass
class LegalChunk:
    """Chunk gotowy do wektoryzacji i wyszukiwania."""
    chunk_id: str
    text: str                       # tekst przekazywany do embeddingu
    units: list[LegalUnit] = field(default_factory=list)  # jednostki wchodzące w skład
    metadata: dict = field(default_factory=dict)

    @property
    def display_reference(self) -> str:
        if not self.units:
            return self.chunk_id
        refs = [u.reference for u in self.units]
        if len(refs) == 1:
            return refs[0]
        return f"{refs[0]} – {refs[-1]}"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LegalDocumentParser:
    """
    Parsuje tekst aktu prawnego i zwraca listę LegalUnit.
    Obsługuje dokumenty z artykułami (ustawa, kodeks) oraz z paragrafami
    (rozporządzenia, statuty).
    """

    def __init__(self, act_title: str = "", source_file: str = ""):
        self.act_title = act_title
        self.source_file = source_file

    # ------------------------------------------------------------------
    # Publiczny interfejs
    # ------------------------------------------------------------------

    def parse(self, text: str) -> list[LegalUnit]:
        text = self._normalize(text)
        if not self.act_title:
            self.act_title = self._detect_title(text)

        # Wybierz strategię w zależności od struktury dokumentu
        has_articles = bool(_RE_ARTICLE.search(text))
        has_paragraphs = bool(_RE_PARA.search(text))

        if has_articles:
            return self._parse_by_articles(text)
        elif has_paragraphs:
            return self._parse_by_paragraphs(text)
        else:
            # Fallback: podział na ustępy / akapity
            return self._parse_flat(text)

    # ------------------------------------------------------------------
    # Normalizacja
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Ujednolica białe znaki i łączniki."""
        # Usuń nadmiarowe spacje w środku linii
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Scal linie rozdzielone miękkim myślnikiem (PDF-owy podział)
        text = re.sub(r"-\n(?=[a-ząćęłńóśźż])", "", text, flags=re.IGNORECASE)
        # Normalizuj znaki końca linii
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Usuń puste linie z samą spacją
        text = re.sub(r"\n[ \t]+\n", "\n\n", text)
        return text

    @staticmethod
    def _detect_title(text: str) -> str:
        m = _RE_TITLE.search(text[:2000])
        if m:
            return " ".join(m.group(0).split()[:12])
        return ""

    # ------------------------------------------------------------------
    # Strategia 1: podział według artykułów (ustawa, kodeks)
    # ------------------------------------------------------------------

    def _parse_by_articles(self, text: str) -> list[LegalUnit]:
        units: list[LegalUnit] = []
        current_section = ""
        preamble_text = ""

        # Znajdź pozycje wszystkich artykułów i nagłówków sekcji
        events: list[tuple[int, str, str]] = []  # (pos, kind, match_text)

        for m in _RE_SECTION.finditer(text):
            events.append((m.start(), "section", m.group(0)))

        for m in _RE_ARTICLE.finditer(text):
            events.append((m.start(), "article", m.group("num")))

        events.sort(key=lambda e: e[0])

        if not events:
            return self._parse_flat(text)

        # Wstęp przed pierwszym artykułem
        first_pos = events[0][0]
        preamble_text = text[:first_pos].strip()
        if preamble_text:
            units.append(LegalUnit(
                unit_type="preamble",
                number="0",
                text=preamble_text,
                act_title=self.act_title,
                source_file=self.source_file,
                char_start=0,
                char_end=first_pos,
            ))

        for i, (pos, kind, value) in enumerate(events):
            end_pos = events[i + 1][0] if i + 1 < len(events) else len(text)
            fragment = text[pos:end_pos].strip()

            if kind == "section":
                current_section = " ".join(fragment.split("\n")[0].split()[:10])
            else:  # article
                units.append(LegalUnit(
                    unit_type="article",
                    number=value,
                    text=fragment,
                    act_title=self.act_title,
                    section=current_section,
                    source_file=self.source_file,
                    char_start=pos,
                    char_end=end_pos,
                ))

        return units

    # ------------------------------------------------------------------
    # Strategia 2: podział według paragrafów (rozporządzenie, statut)
    # ------------------------------------------------------------------

    def _parse_by_paragraphs(self, text: str) -> list[LegalUnit]:
        units: list[LegalUnit] = []
        current_section = ""

        events: list[tuple[int, str, str]] = []
        for m in _RE_SECTION.finditer(text):
            events.append((m.start(), "section", m.group(0)))
        for m in _RE_PARA.finditer(text):
            events.append((m.start(), "paragraph", m.group("num")))
        events.sort(key=lambda e: e[0])

        if not events:
            return self._parse_flat(text)

        first_pos = events[0][0]
        preamble = text[:first_pos].strip()
        if preamble:
            units.append(LegalUnit(
                unit_type="preamble", number="0", text=preamble,
                act_title=self.act_title, source_file=self.source_file,
                char_start=0, char_end=first_pos,
            ))

        for i, (pos, kind, value) in enumerate(events):
            end_pos = events[i + 1][0] if i + 1 < len(events) else len(text)
            fragment = text[pos:end_pos].strip()

            if kind == "section":
                current_section = " ".join(fragment.split("\n")[0].split()[:10])
            else:
                units.append(LegalUnit(
                    unit_type="paragraph",
                    number=value,
                    text=fragment,
                    act_title=self.act_title,
                    section=current_section,
                    source_file=self.source_file,
                    char_start=pos,
                    char_end=end_pos,
                ))

        return units

    # ------------------------------------------------------------------
    # Strategia 3: podział płaski (fallback)
    # ------------------------------------------------------------------

    def _parse_flat(self, text: str) -> list[LegalUnit]:
        """Dzieli na akapity, gdy brak struktury artykuł/paragraf."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        units = []
        for i, p in enumerate(paragraphs):
            units.append(LegalUnit(
                unit_type="section",
                number=str(i + 1),
                text=p,
                act_title=self.act_title,
                source_file=self.source_file,
            ))
        return units


# ---------------------------------------------------------------------------
# Chunker – łączy jednostki w chunk-i gotowe do wektoryzacji
# ---------------------------------------------------------------------------

class LegalChunker:
    """
    Łączy LegalUnit-y w LegalChunk-i o kontrolowanym rozmiarze.

    Zasady:
    - Nigdy nie dziel pojedynczego artykułu/paragrafu w środku.
    - Jeśli artykuł jest dłuższy niż max_chars → podziel na ustępy.
    - Overlap: ostatnia jednostka poprzedniego chunka jest pierwszą kolejnego.
    - Każdy chunk zawiera pełny kontekst (tytuł aktu, rozdział).
    """

    def __init__(
        self,
        max_chars: int = 2000,
        overlap_units: int = 1,
        min_chars: int = 100,
    ):
        self.max_chars = max_chars
        self.overlap_units = overlap_units
        self.min_chars = min_chars

    def chunk(self, units: list[LegalUnit]) -> list[LegalChunk]:
        if not units:
            return []

        chunks: list[LegalChunk] = []
        buffer: list[LegalUnit] = []
        buffer_len = 0
        chunk_idx = 0

        def flush(buf: list[LegalUnit]) -> LegalChunk:
            nonlocal chunk_idx
            combined = "\n\n".join(u.full_context_text for u in buf)
            c = LegalChunk(
                chunk_id=f"chunk_{chunk_idx:04d}",
                text=combined,
                units=list(buf),
                metadata={
                    "act_title": buf[0].act_title,
                    "section": buf[0].section,
                    "first_ref": buf[0].reference,
                    "last_ref": buf[-1].reference,
                    "unit_count": len(buf),
                },
            )
            chunk_idx += 1
            return c

        for unit in units:
            unit_text = unit.full_context_text
            unit_len = len(unit_text)

            # Artykuł przekracza max_chars → podziel na ustępy
            if unit_len > self.max_chars:
                # Najpierw opróżnij bufor
                if buffer:
                    chunks.append(flush(buffer))
                    buffer = buffer[-self.overlap_units:] if self.overlap_units else []
                    buffer_len = sum(len(u.full_context_text) for u in buffer)

                # Podziel duży artykuł na ustępy
                for sub in self._split_large_unit(unit):
                    chunks.append(LegalChunk(
                        chunk_id=f"chunk_{chunk_idx:04d}",
                        text=sub,
                        units=[unit],
                        metadata={
                            "act_title": unit.act_title,
                            "section": unit.section,
                            "first_ref": unit.reference,
                            "last_ref": unit.reference,
                            "unit_count": 1,
                            "is_split": True,
                        },
                    ))
                    chunk_idx += 1
                continue

            # Normalny artykuł – dodaj do bufora lub zacznij nowy
            if buffer and buffer_len + unit_len > self.max_chars:
                chunks.append(flush(buffer))
                # Overlap: zacznij nowy chunk od ostatnich N jednostek
                buffer = buffer[-self.overlap_units:] if self.overlap_units else []
                buffer_len = sum(len(u.full_context_text) for u in buffer)

            buffer.append(unit)
            buffer_len += unit_len

        # Reszta
        if buffer and buffer_len >= self.min_chars:
            chunks.append(flush(buffer))

        return chunks

    def _split_large_unit(self, unit: LegalUnit) -> Iterator[str]:
        """Dzieli duży artykuł na ustępy (ust. 1., ust. 2., …)."""
        text = unit.full_context_text
        # Spróbuj podzielić na ustępy
        splits = re.split(r"(?m)(?=^\d+\.\s)", text)
        splits = [s.strip() for s in splits if s.strip()]

        if len(splits) <= 1:
            # Nie ma ustępów → podział mechaniczny z zachowaniem zdań
            yield from self._split_by_sentences(text)
            return

        buf = ""
        for fragment in splits:
            if len(buf) + len(fragment) > self.max_chars and buf:
                yield f"[{unit.act_title}] {unit.reference} (cd.)\n" + buf.strip()
                buf = fragment + "\n"
            else:
                buf += fragment + "\n"
        if buf.strip():
            yield f"[{unit.act_title}] {unit.reference} (cd.)\n" + buf.strip()

    def _split_by_sentences(self, text: str) -> Iterator[str]:
        """Ostateczny fallback: podział na zdania."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) > self.max_chars and buf:
                yield buf.strip()
                buf = s + " "
            else:
                buf += s + " "
        if buf.strip():
            yield buf.strip()


# ---------------------------------------------------------------------------
# Funkcja wysokiego poziomu
# ---------------------------------------------------------------------------

def chunk_legal_document(
    text: str,
    source_file: str = "",
    act_title: str = "",
    max_chars: int = 2000,
    overlap_units: int = 1,
) -> list[LegalChunk]:
    """
    Główna funkcja do wywołania z aplikacji.

    Parameters
    ----------
    text        : surowy tekst aktu prawnego
    source_file : nazwa pliku źródłowego (do metadanych)
    act_title   : tytuł aktu (jeśli znany; wykrywany automatycznie)
    max_chars   : maksymalna długość chunk-a w znakach
    overlap_units : liczba jednostek powtarzanych między chunk-ami

    Returns
    -------
    Lista LegalChunk gotowych do wektoryzacji.
    """
    parser = LegalDocumentParser(act_title=act_title, source_file=source_file)
    units = parser.parse(text)
    chunker = LegalChunker(max_chars=max_chars, overlap_units=overlap_units)
    return chunker.chunk(units)


# ---------------------------------------------------------------------------
# Szybki test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """
USTAWA z dnia 23 kwietnia 1964 r. – Kodeks cywilny

CZĘŚĆ OGÓLNA

TYTUŁ I
Przepisy wstępne

Art. 1. Kodeks niniejszy reguluje stosunki cywilnoprawne między osobami fizycznymi
i osobami prawnymi.

Art. 2. (uchylony)

Art. 3. Ustawa nie ma mocy wstecznej, chyba że to wynika z jej brzmienia lub celu.

ROZDZIAŁ 2
Osoby fizyczne

Art. 8.
1. Każdy człowiek od chwili urodzenia ma zdolność prawną.
2. (uchylony)

Art. 9. W razie urodzenia się dziecka domniemywa się, że przyszło ono na świat żywe.

Art. 10.
1. Pełnoletnim jest, kto ukończył lat osiemnaście.
2. Przez zawarcie małżeństwa małoletni uzyskuje pełnoletność. Nie traci jej
w razie unieważnienia małżeństwa.
"""

    chunks = chunk_legal_document(SAMPLE, source_file="kc.pdf", max_chars=600)
    print(f"Liczba chunk-ów: {len(chunks)}\n")
    for c in chunks:
        print(f"{'='*60}")
        print(f"ID: {c.chunk_id} | Odwołanie: {c.display_reference}")
        print(f"Metadane: {c.metadata}")
        print(f"Tekst ({len(c.text)} znaków):\n{c.text[:300]}{'...' if len(c.text)>300 else ''}")
        print()
