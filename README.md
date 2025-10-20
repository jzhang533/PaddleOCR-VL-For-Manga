## SFT PaddleOCR-VL-0.9B

WARNING: the script is being used to overfit the 6 images in data folder

### SFT

```bash
sh run.sh
``` 

### use it 

```python
python use_transformers.py
```

### expected output

User: OCR:
Assistant: 素直にあやまるしかあやまるわよ！ ..........<repeating>..........

### Future

probablly will SFT it on a full dataset
