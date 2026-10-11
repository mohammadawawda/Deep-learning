from utils import *

# collect raw peptides and labels
all_peps, all_labels = [], []
for fn in os.listdir(DATA_DIR):
    if not fn.endswith(".txt"):
        continue
    ps, ls = parse_file(os.path.join(DATA_DIR, fn))
    all_peps.extend(ps)
    all_labels.extend(ls)

# convert peptides to index sequences (N,9)
pep_idx = torch.tensor(
    [[AA_DICT[aa] for aa in pep] for pep in all_peps],
    dtype=torch.long)

# convert labels to tensor (N,6)
labels = torch.tensor(all_labels, dtype=torch.float32)

# 90% train and 10% test
dataset = TensorDataset(pep_idx, labels)
n_train = int(0.9 * len(dataset))
train_ds, test_ds = random_split(dataset, [n_train, len(dataset) - n_train])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# a simple MLP that keeps the input size
my_model = SimpleClassifier(num_inputs=9, num_hidden=9,
                            num_outputs=6)

my_model.to(DEVICE)
loss_func = nn.MSELoss()
sgd_optimizer = torch.optim.SGD(my_model.parameters(), lr=LR)

part1 = "SimpleClassifier Training & Validation Loss"
print(part1)
train_losses, test_losses = train_model_epoch_eval(my_model, sgd_optimizer, train_loader, test_loader,
                       loss_func, EPOCHS, DEVICE)
plot_loss_curve(train_losses, test_losses, part1,
                "Epoch", "MSE Loss")

# the better model with embeddings
model = PeptideMLP().to(DEVICE)

# pos_weight gives the positives more weight because there are many more negatives
neg_counts = (labels == 0).sum(dim=0)
pos_counts = (labels == 1).sum(dim=0)
pos_weight = (neg_counts.float() / (pos_counts.float() + 1e-8)).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

train_losses, test_losses = [], []

part2 = "PeptideMLP Training & Validation Loss"
print(part2)

for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    running_loss, total = 0.0, 0
    for pep_batch, y_batch in train_loader:
        pep_batch = pep_batch.to(DEVICE)
        y_batch   = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(pep_batch)               # (B,6)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pep_batch.size(0)
        total        += pep_batch.size(0)
    train_losses.append(running_loss / total)

    # test
    model.eval()
    running_loss, total = 0.0, 0
    with torch.no_grad():
        for pep_batch, y_batch in test_loader:
            pep_batch = pep_batch.to(DEVICE)
            y_batch   = y_batch.to(DEVICE)
            logits    = model(pep_batch)
            loss      = criterion(logits, y_batch)

            running_loss += loss.item() * pep_batch.size(0)
            total        += pep_batch.size(0)
    test_losses.append(running_loss / total)

    print(f"Epoch {epoch:02d} — train_loss: {train_losses[-1]:.4f}, "
          f"test_loss:  {test_losses[-1]:.4f}")

plot_loss_curve(train_losses, test_losses, part2, "Epoch",
                "BCEWithLogits Loss")


# the same model without the non linear parts
model = LinearPeptideMLP().to(DEVICE)

neg_counts = (labels == 0).sum(dim=0)
pos_counts = (labels == 1).sum(dim=0)
pos_weight = (neg_counts.float() / (pos_counts.float() + 1e-8)).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

train_losses, test_losses = [], []
part3 = "Linear-Only PeptideMLP Loss"
print(part3)
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    running_loss, total = 0.0, 0
    for pep_batch, y_batch in train_loader:
        pep_batch, y_batch = pep_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(pep_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * pep_batch.size(0)
        total        += pep_batch.size(0)
    train_losses.append(running_loss/total)

    # test
    model.eval()
    running_loss, total = 0.0, 0
    with torch.no_grad():
        for pep_batch, y_batch in test_loader:
            pep_batch, y_batch = pep_batch.to(DEVICE), y_batch.to(DEVICE)
            logits    = model(pep_batch)
            loss      = criterion(logits, y_batch)
            running_loss += loss.item() * pep_batch.size(0)
            total        += pep_batch.size(0)
    test_losses.append(running_loss/total)
    print(f"Epoch {epoch:02d} — train_loss: {train_losses[-1]:.4f}, "
          f"test_loss:  {test_losses[-1]:.4f}")

plot_loss_curve(train_losses, test_losses, part3,
                "Epoch", "BCEWithLogits Loss")


# e) the 3 spike peptides with the highest probability
def load_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.append(line.strip())
    return "".join(seq)

spike = load_fasta("P0DTC2.fasta")
assert len(spike) == 1273

# every 9 amino acids in a row
peptides = [spike[i:i+9] for i in range(len(spike)-8)]

# as indices, shape (N,9)
pep_idxs = torch.tensor(
    [[AA_DICT[aa] for aa in pep] for pep in peptides],
    dtype=torch.long,
    device=DEVICE
)

model.eval()
with torch.no_grad():
    logits = model(pep_idxs)           # shape (N,6)
    probs  = torch.sigmoid(logits)     # shape (N,6)

# allele names by index
alleles = [None]*6
for name, idx in ALLELES_KEY.items():
    alleles[idx] = name

# the 3 highest probabilities over all peptides and alleles
flat = probs.cpu().numpy().flatten()
top3 = flat.argsort()[-3:][::-1]

print("Top 3 Spike 9-mers and alleles:")
for idx in top3:
    pep_i     = idx // 6
    allele_i  = idx % 6
    p         = flat[idx]
    print(f"  Peptide: {peptides[pep_i]}  Allele: {alleles[allele_i]}  Probability: {p:.4f}")

