import torch
import model
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda"

video_model = model.VideoRec()
video_model.to(device)
optimizer = torch.optim.Adam(video_model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()


batch_size = 64
import get_data
train_dataset = get_data.MyDataset(get_data.image_list,get_data.label_list)
loader = DataLoader(train_dataset,batch_size=batch_size,pin_memory = True,shuffle=True,num_workers=0)

for epoch in (range(1024)):
    pbar = tqdm(loader,total=len(loader))
    for token_inp,token_tgt in pbar:
        token_inp = token_inp.to(device)
        token_tgt = token_tgt.to(device)

        logits = video_model(token_inp)
        loss = criterion(logits.view(-1,logits.size(-1)),token_tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_correct = (torch.argmax(logits, dim=-1) == (token_tgt)).type(torch.float).sum().item() / len(token_inp)

        pbar.set_description(f"epoch:{epoch + 1}, train_loss:{loss.item():.5f}, train_correct:{train_correct:.5f}")


