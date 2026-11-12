# Wanneer is een label gekend?

Een trainingslabel mag alleen gebruikt worden als de afloop gekend was op het moment dat het model
gebruikt wordt. Een datum in het verleden is niet genoeg.

| Label | Gekend op | Gevolg voor de split |
|---|---|---|
| geannuleerd na T | de annulatiedatum, ten laatste de aankomst | gekend de dag van aankomst |
| no-show | de dag na de aankomst | gekend de dag na aankomst |
| uitgecheckt | bij aankomst weet je al dat het geen annulatie of no-show is | gekend de dag na aankomst |

Het eerste beslismoment van de validatie is 2 december 2016 (voor de nacht van 1 januari). Op dat
moment is de afloop gekend van elke boeking met aankomst tot en met 30 november. Daarom stopt de
training op 30-11-2016 en de hertraining op 28-02-2017 (embargo = horizon + 1 dag, `grenzen()` in
`src/train.py`, gecontroleerd in `tests/test_split.py`).

**Beperking.** Binnen de validatie wordt het model niet bijgewerkt: voor de nacht van 31 maart
beslist het met de kennis van 30 november, niet met die van 1 maart. Dat maakt de validatie eerder
strenger dan de praktijk, niet optimistischer.

De features en hun beschikbaarheid staan in [feature_beschikbaarheid.csv](feature_beschikbaarheid.csv).
De kolom `bekend_op_T` zegt of de waarde op het beslismoment bestaat; `kan_later_wijzigen` zegt of
de export een latere waarde kan tonen. 20 % van de actieve boekingen heeft minstens één wijziging
(`booking_changes > 0`), en wanneer die gebeurde staat niet in de export. Voor die boekingen kan de
waarde in de export afwijken van die op T. Dat is niet te herstellen met deze export, en staat
daarom als beperking in het rapport.
