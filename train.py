import torch
from transformers import AutoTokenizer
from model import ClimateClassifier
from loss import WeightedBCELoss


MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = ClimateClassifier(
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS
)

model.to(device)


pos_weight = torch.ones(NUM_LABELS).to(device)

criterion = WeightedBCELoss(
    pos_weight=pos_weight
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


def train_step(batch):

    model.train()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    loss = criterion(logits, labels)

    loss.backward()

    optimizer.step()

    return loss.item()


print("Training pipeline initialized.")
