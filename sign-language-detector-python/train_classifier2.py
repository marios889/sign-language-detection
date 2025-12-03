import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import pickle

# 1) model
class SoftmaxClassification(nn.Module):
    def __init__(self,num_input_features):
        super(SoftmaxClassification,self).__init__()
        self.n_features_in_ = num_input_features
        self.model = nn.Sequential(
            nn.Linear(num_input_features,24),
            nn.ReLU(),
            nn.Linear(24,12),
        )

    def forward(self,x):
        output = self.model(x)
        _,y_predicted = torch.max(torch.softmax(self.model(x),dim=1),1)
        return output,y_predicted


if __name__ == "__main__":
    # 0) data preprocessing
    data_dict = pickle.load(open('./data.pickle', 'rb'))

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    n_samples, n_features = data.shape
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    # y_train = y_train.view(y_train.shape[0])
    y_train = y_train.to(torch.long)
    # y_test = y_test.view(y_test.shape[0])
    y_test = y_test.to(torch.long)

    # 2) loss and optimizer
    model = SoftmaxClassification(n_features)
    learning_rate = 0.4
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    # 3) Training
    n_iterations = 1000
    for epoch in range(n_iterations):
        # forward and loss calculation
        out,_ = model(x_train)

        loss_value = loss_fn(out,y_train)
        # backward gradient
        loss_value.backward()
        # optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) %20 ==0:
            print(f"{epoch+1} loss : {loss_value.item():.4f}")

    _,y_predicted_test = model(x_test)
    score = accuracy_score(y_predicted_test, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    f = open('deepmodel.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()




