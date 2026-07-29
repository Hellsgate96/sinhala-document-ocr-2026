# print_photos - held-out real eval pages

Real photographs of printed Sinhala text used **only for evaluation** (never
mixed into training). End-to-end scored with `scripts/run_realistic_eval.py`.

| File | Description | Source / licence |
|------|-------------|------------------|
| `page_poem_print.jpg` | Photo of a printed poem page ("හමුවෙලා හදවතේ...", 8 poem lines + small footer "පන්තියේ අන්තිමයා") in a traditional serif book font on grey paper. | Supplied by the project owner (own photo of a shared social-media poem card), Jul 2026. Used with permission for research evaluation. |
| `page_song_lyrics.jpg` | Low-resolution lyrics-site image ("සිංහල ජය ගීත හඬට", 22 lyric lines + tiny credits footer); every line is ~15-20 px tall — the tiny-text stress case. | Supplied by the project owner (Jul-28 notebook test image, lyrics card shared on social media). Used with permission for research evaluation. |

Ground truth was transcribed manually from 4-8x zoomed line crops
(`page_poem_print`: note the long-uu "ූ" endings; `page_song_lyrics`: the
site prints ද where standard spelling has ඳ in "බැදුණේ"/"සිදුණේ" — GT follows
the printed image, and keeps the "...//" refrain notation).
