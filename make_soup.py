import torch
cks = ["ck_ep03.pth", "ck_ep04.pth", "ck_ep06.pth"]
sd = None
for c in cks:
    s = torch.load(c, map_location="cpu", weights_only=True)
    if sd is None:
        sd = {k: v.clone().float() / len(cks) for k, v in s.items()}
    else:
        for k in sd:
            sd[k] += s[k].float() / len(cks)
torch.save(sd, "soup.pth")
print("saved soup.pth from", cks)
