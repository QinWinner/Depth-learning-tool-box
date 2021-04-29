criterion = torch.nn.CrossEntropyLoss()
net = model_zoo.resnet50(pretrained=True)
net.fc = nn.Linear(2048, 120)

with torch.cuda.device(0):
   net = net.cuda()

basic_optim = torch.optim.SGD(net.parameters(), lr=1e-5)
optimizer = ScheduledOptim(basic_optim)


lr_mult = (1 / 1e-5) ** (1 / 100)
lr = []
losses = []
best_loss = 1e9
for data, label in train_data:
    with torch.cuda.device(0):
        data = Variable(data.cuda())
        label = Variable(label.cuda())
    # forward
    out = net(data)
    loss = criterion(out, label)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr.append(optimizer.learning_rate)
    losses.append(loss.data[0])
    optimizer.set_learning_rate(optimizer.learning_rate * lr_mult)
    if loss.data[0] < best_loss:
        best_loss = loss.data[0]
    if loss.data[0] > 4 * best_loss or optimizer.learning_rate > 1.:
        break

plt.figure()
plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.plot(np.log(lr), losses)
plt.show()
plt.figure()
plt.xlabel('num iterations')
plt.ylabel('learning rate')
plt.plot(lr)