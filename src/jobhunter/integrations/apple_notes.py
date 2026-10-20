"""macOS Apple Notes bridge via JXA (``osascript -l JavaScript``).

* Idempotent upsert: finds the note by title inside the folder (creating the folder if needed) and
  replaces its body, so re-running the pipeline the same day updates — never duplicates — the note.
* The HTML body is passed through a temp file, not argv: no shell-quoting bugs, no ARG_MAX limits.
* Notes derives a note's title from the first line of its body, so the body must start with an
  ``<h1>`` equal to the title — ``render`` guarantees that.
* First run triggers macOS's Automation permission prompt ("Terminal wants to control Notes"); the
  script blocks until it's answered, so a timeout says to look for that dialog.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from jobhunter.telemetry import get_logger, span

log = get_logger("notes")

JXA_UPSERT = r"""
ObjC.import('Foundation');
function readUtf8(path) {
  const s = $.NSString.stringWithContentsOfFileEncodingError($(path), $.NSUTF8StringEncoding, null);
  if (!s) throw new Error('cannot read ' + path);
  return ObjC.unwrap(s);
}
function run(argv) {
  const [folderName, title, htmlPath, accountName] = argv;
  const Notes = Application('Notes');
  const account = accountName ? Notes.accounts.byName(accountName) : Notes.defaultAccount();
  let folder;
  const found = account.folders.whose({ name: folderName });
  if (found.length > 0) {
    folder = found[0];
  } else {
    folder = Notes.Folder({ name: folderName });
    account.folders.push(folder);
    folder = account.folders.byName(folderName);
  }
  const body = readUtf8(htmlPath);
  const existing = folder.notes.whose({ name: title });
  if (existing.length > 0) {
    const note = existing[0];
    note.body = body;
    return 'updated:' + note.id();
  }
  const note = Notes.Note({ body: body });
  folder.notes.push(note);
  try { return 'created:' + note.id(); }
  catch (e) { return 'created:' + folder.notes.whose({ name: title })[0].id(); }
}
"""


@dataclass
class NoteResult:
    action: str  # created | updated
    note_id: str


class AppleNotesError(RuntimeError):
    pass


class AppleNotes:
    def __init__(
        self, folder: str = "Job Search", account: str | None = None, timeout: float = 120.0
    ) -> None:
        self.folder = folder
        self.account = account
        self.timeout = timeout

    @staticmethod
    def available() -> bool:
        return sys.platform == "darwin" and Path("/usr/bin/osascript").exists()

    def upsert(self, title: str, html_body: str) -> NoteResult:
        if not self.available():
            raise AppleNotesError("Apple Notes is only available on macOS")
        # Launch Notes first: a cold start on a busy Mac is part of what used to blow the timeout.
        with contextlib.suppress(subprocess.SubprocessError, OSError):
            subprocess.run(["open", "-ga", "Notes"], capture_output=True, timeout=15, check=False)
        with span("notes.upsert", title=title), tempfile.TemporaryDirectory() as tmp:
            body_path = Path(tmp) / "body.html"
            body_path.write_text(html_body, encoding="utf-8")
            script_path = Path(tmp) / "upsert.js"
            script_path.write_text(JXA_UPSERT, encoding="utf-8")
            try:
                proc = self._osascript(script_path, title, body_path)
            except subprocess.TimeoutExpired:
                raise AppleNotesError(
                    f"Notes didn't answer within {self.timeout:.0f} s. If macOS shows a dialog "
                    '"… wants to control Notes", click OK (or allow it in System Settings › Privacy & '
                    "Security › Automation), then run `jh publish` to send today's digest again."
                ) from None
        if proc.returncode != 0:
            err = proc.stderr.strip()
            if "-1743" in err or "Not authorized" in err:
                err += (
                    "\n→ Grant access: System Settings › Privacy & Security › Automation › "
                    "<your terminal> › Notes"
                )
            raise AppleNotesError(err)
        action, _, note_id = proc.stdout.strip().partition(":")
        log.info("Apple Note {} ({}): {}", action, self.folder, title)
        return NoteResult(action, note_id)

    def _osascript(self, script: Path, title: str, body: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "/usr/bin/osascript",
                "-l",
                "JavaScript",
                str(script),
                self.folder,
                title,
                str(body),
                self.account or "",
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
