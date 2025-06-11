# geo_akinator

We extend the traditional geocoding task by introducing a novel challenge: predicting the geographic coordinates (latitude and longitude) of a location based solely on a natural language description. We built a model capable of interpreting detailed geographic descriptions and returning plausible coordinates. We collected a new dataset of over 800,000 samples and trained a lightweight BERT-based model to outperform prior art


For model inference you can use our demo, deployed with Stremlit.
```
pip3 install streamlit
```

```
streamlit run app.py
```
