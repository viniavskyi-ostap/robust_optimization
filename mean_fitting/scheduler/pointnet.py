import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """Feature transformation prediction sub-network"""

    def __init__(self, input_dim, output_dim):
        super(STN, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim * output_dim)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1, self.output_dim, self.output_dim)
        iden = torch.eye(self.output_dim, device=x.device)
        x = x + iden
        return x


class PointNetEncoder(nn.Module):
    """Encodes set of N points with a high-dimensional vector in a permutation invariant way"""
    def __init__(self, global_feat=True, feature_transform=False,
                 input_dim=3, point_dim=3):
        super(PointNetEncoder, self).__init__()
        if input_dim < point_dim:
            raise ValueError('point dimension must be smaller than overall input dim')
        self.point_dim = point_dim

        self.stn = STN(input_dim, point_dim)
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STN(input_dim=64, output_dim=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)

        if D > self.point_dim:
            x, feature = x[..., :self.point_dim], x[..., self.point_dim:]
        x = torch.bmm(x, trans)
        if D > self.point_dim:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).expand(-1, -1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetClassifier(nn.Module):
    def __init__(self, output_dim=1, input_dim=3, point_dim=3):
        super(PointNetClassifier, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, input_dim=input_dim,
                                    point_dim=point_dim)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat


class PointNetSegmentation(nn.Module):
    def __init__(self, output_dim=1, input_dim=3, point_dim=3):
        super(PointNetSegmentation, self).__init__()
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, input_dim=input_dim,
                                    point_dim=point_dim)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x, trans_feat


if __name__ == '__main__':
    x = torch.randn(3, 5, 1024)
    model = PointNetClassifier(output_dim=1, input_dim=5, point_dim=2)
    y = model(x)
    print(y[0].shape, y[1].shape)

    x = torch.randn(3, 5, 1024)
    model = PointNetSegmentation(output_dim=1, input_dim=5, point_dim=2)
    y = model(x)
    print(y[0].shape, y[1].shape)
