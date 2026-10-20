**AI/ML Engineer — Jan Kowalski**

Dzień dobry,

aplikuję na stanowisko AI/ML Engineer w Acme Analytics. Z ogłoszenia wynika, że budujecie asystenta opartego na RAG dla zespołów analitycznych i szukacie osoby, która przeprowadzi go od prototypu do produkcji. Dokładnie tym zajmowałem się przez ostatnie półtora roku.

W projekcie DocuMind zbudowałem asystenta RAG nad 40 tys. dokumentów wewnętrznych klienta z branży fintech. Hybrydowe wyszukiwanie BM25 z pgvector oraz streaming odpowiedzi obniżyły opóźnienie p95 z 9,1 s do 2,4 s. Każda zmiana promptu lub modelu przechodziła przez offline'owy zestaw ewaluacyjny (RAGAS i 300 oznaczonych par pytanie–odpowiedź) uruchamiany w CI, więc regresje wychwytywaliśmy przed wdrożeniem, a nie po zgłoszeniach użytkowników.

Wasz stos (Python, FastAPI, PostgreSQL, AWS) pokrywa się z tym, na czym pracuję na co dzień. Przy Pricing API utrzymywałem 1,8 tys. zapytań na sekundę przy p99 poniżej 60 ms, a migrację endpointu z Django do asynchronicznego FastAPI przeprowadziłem bez przestoju, z użyciem shadow traffic i feature flag. Chętnie przeniosę te praktyki do Acme Analytics: mierzalną jakość odpowiedzi modelu, przewidywalne opóźnienia i bezpieczne wdrożenia.

Będzie mi miło porozmawiać o tym, jak mogę pomóc w rozwoju Waszego asystenta.

Z pozdrowieniami,

Jan Kowalski  
jan.kowalski@example.com · +48 600 000 000 · https://www.linkedin.com/in/jan-kowalski · https://github.com/jankowalski
