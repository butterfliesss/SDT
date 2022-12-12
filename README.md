# A Transformer-based Model with Self-distillation for Multimodal Emotion Recognition in Conversations
This repository is the implement of our paper A Transformer-based Model with Self-distillation for Multimodal Emotion Recognition in Conversations.

## Model Architecture
<!-- ![Image of SDT](fig/SDT.jpg) -->
<div align="center">
    <img src="fig/SDT.jpg" width="85%" title="SDT."</img>
</div>

## Setup
- Check the packages needed or simply run the command:
```console

pip install -r requirements.txt
```
- Download the preprocessed datasets from [here](https://drive.google.com/drive/folders/1J1mvbqQmVodNBzbiOIxRiWOtkP6qqP-K?usp=sharing), and put them into `data/`.

## Run SDT model
- Run the model on IEMOCAP dataset:
```console

bash exec_iemocap.sh
```
- Run the model on MELD dataset:
```console

bash exec_meld.sh
```

## Acknowledgements
- Special thanks to the [COSMIC](https://github.com/declare-lab/conv-emotion) and [MMGCN](https://github.com/hujingwen6666/MMGCN) for sharing their codes and datasets.