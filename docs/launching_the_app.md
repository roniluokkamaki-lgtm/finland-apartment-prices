# Laskurin käynnistäminen

## Pikakäynnistys

Aja projektin juurikansiossa:

```bash
python run.py
```

Tämä käynnistää palvelimen ja avaa laskurin automaattisesti selaimeen osoitteessa `http://localhost:8501`.

Sulje painamalla `Ctrl+C` terminaalissa.

---

## Vaatimukset

- Python 3.9+
- Paketit: `streamlit`, `pandas`, `numpy`, `statsmodels`

Asenna puuttuvat paketit:

```bash
pip install streamlit pandas numpy statsmodels
```

## Huomio

Malli täytyy kouluttaa ennen ensimmäistä käynnistystä (tai jos data päivittyy):

```bash
python app/model.py
```

Tämä luo tiedoston `models/model2_fe.json`, jota laskuri käyttää hintojen laskemiseen.
