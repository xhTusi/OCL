import os
import os.path as osp
import pdb
import numpy as np
import torch
from requests import options
from torch.autograd import Variable
import torch.nn.functional as F
from IP_test import args
from core import evaluation
# results, pred, label = test.test(net, criterion, full_testloader, testloader, outloader, logits_min, dis_min,loss_r, unknow, epoch=epoch, **options)
def CACLoss(distances, labels):
    '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
    true = torch.gather(distances, 1, labels.view(-1, 1)).view(-1)
    non_gt = torch.Tensor([[i for i in range(9) if labels[x] != i] for x in range(21)]).long().cuda()
    others = torch.gather(distances, 1, non_gt)
    anchor = torch.mean(true)
    tuplet = torch.exp(-others + true.unsqueeze(1))
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))
    total = 0.1 * anchor + tuplet
    return total, anchor, tuplet

def val(net, full_testloader,   unknow, **options):
    global best_acc
    global best_anchor
    global best_cac
    net.eval()
    _pred_k, _pred_u, _pred_all, _labels = [], [], [], []
    correct_u, total_u = 0, 0
    correct_all, total_all = 0, 0
    torch.cuda.empty_cache()#todo 用于清除 GPU 缓存
    anchor_loss = 0
    cac_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(full_testloader):
            if options['use_gpu']:
                inputs, labels = inputs.cuda(), labels.cuda()
            # targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)

            outputs = net(inputs)
            # pdb.set_trace()
            labels = labels.to(torch.long)

            cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], labels)

            anchor_loss += anchorLoss
            cac_loss += cacLoss

            _, predicted = outputs[1].min(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            # progress_bar(batch_idx, len(full_testloader), 'Acc: %.3f%% (%d/%d)'
            #              % (100. * correct / total, correct, total))
            _labels.append(labels.data.cpu().numpy())

    anchor_loss /= len(full_testloader)
    cac_loss /= len(full_testloader)
    acc = 100. * correct / total



    # Save checkpoint.
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': 000,
    }
    if not os.path.isdir('networks/weights/{}'.format(args.dataset)):
        os.mkdir('networks/weights/{}'.format(args.dataset))
    if args.dataset == 'IP':
        if not os.path.isdir('networks/weights/IP'):
            os.mkdir('networks/weights/IP')
    save_name = '{}_{}_{}CACclassifier'.format(args.dataset, args.trial, args.name)
    if anchor_loss <= best_anchor:
        print('Saving..')
        torch.save(state, 'networks/weights/{}/'.format(args.dataset) + save_name + 'AnchorLoss.pth')
        best_anchor = anchor_loss

        if args.dataset == 'IP':
            save_name = save_name.replace('+10', '+50')
            torch.save(state, 'networks/weights/CIFAR+50/' + save_name + 'AnchorLoss.pth')

    if cac_loss <= best_cac:
        print('Saving..')
        torch.save(state, 'networks/weights/{}/'.format(args.dataset) + save_name + 'CACLoss.pth')
        best_cac = cac_loss
        if args.dataset == 'IP':
            save_name = save_name.replace('+10', '+50')
            torch.save(state, 'networks/weights/CIFAR+50/' + save_name + 'CACLoss.pth')

    if acc >= best_acc:
        print('Saving..')
        torch.save(state, 'networks/weights/{}/'.format(args.dataset) + save_name + 'Accuracy.pth')
        best_acc = acc

        if args.dataset == 'IP':
            save_name = save_name.replace('+10', '+50')
            torch.save(state, 'networks/weights/CIFAR+50/' + save_name + 'Accuracy.pth')
    #
    # if args.tensorboard:
    #     writer.add_scalar('val/accuracy', acc, epoch)
    #     writer.add_scalar('val/anchorLoss', anchor_loss, epoch)
    #     writer.add_scalar('val/CACLoss', cac_loss, epoch)

# def test(net, criterion,full_testloader, testloader, outloader,logits_min,dis_min,loss_r,unknow, epoch=None, **options):
#     net.eval()#todo 将神经网络设置为评估模式
#
#     correct, total = 0, 0
#
#     correct_u, total_u = 0, 0
#
#     correct_all, total_all = 0, 0
#
#     torch.cuda.empty_cache()#todo 用于清除 GPU 缓存
#
#     _pred_k, _pred_u, _pred_all, _labels = [],[],[], []
#
#     with torch.no_grad():#todo 创建一个没有梯度计算的上下文，以便在评估模式下进行前向传播，测试的时候都需要这样
#         for data, labels in testloader:#todo 循环遍历测试数据集中的每个批次
#             if options['use_gpu']:#todo 检查是否使用 GPU
#                 data, labels = data.cuda(), labels.cuda()#labels都是已知的标签
#
#             with torch.set_grad_enabled(False):
#                 data = data.squeeze(1).permute(0, 3, 1, 2)#todo 调整输入数据的维度
#                 x, y = net(data, True)#todo 进行前向传播，获取模型的输出，x.mean(1),self.head(x.mean(1))
#
#                 logits, dis, radius = criterion(x, y)#todo 计算模型的输出和相关指标[(21X8),(21X1),(1X1)]
#                 predictions = logits.data.max(1)[1]#todo 获取预测结果(最大值所对应的索引值)，.max(1)[1]中的方括号[1]则表示返回最大值的索引
#                 #for i in range(predictions.shape[0]):todo max(1)中的1表示按照第一个维度（行）求最大值，就是每一行最大值的索引值
#                     #print('known:'+str(dis[i]))#
#                 total += labels.size(0)#todo  size(0) 通常是用于计算张量（Tensor）的第一个维度的大小。就是batch_size的大小（21）
#                 correct += (predictions == labels.data).sum()#todo 计算预测正确的个数
#
#                 _pred_k.append(logits.data.cpu().numpy())#todo 预测的结果保存起来(对应的已知类精度)
#                 _labels.append(labels.data.cpu().numpy())#todo 对应正确的标签一起存储
#
#         for batch_idx, (data, labels) in enumerate(outloader):
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()#todo labels全部等于8
#
#             with torch.set_grad_enabled(False):
#
#                 data = data.squeeze(1).permute(0, 3, 1, 2)#todo [21,1,7,7,200]->[21,200,7,7]
#                 x, y = net(data, True)#todo [21,64],[21,8]
#                 # x, y = net(data, return_feature=True)
#                 logits, dis, radius = criterion(x, y)
#                 predictions = logits.data.max(1)[1]#todo 得到的是每一行最大值的索引值（对应的标签）
#
#                 total_u += labels.size(0)
#
#                 for i in range(predictions.shape[0]):
#                     #print('unknown:'+str(dis[i]))
#                     #if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
#                     #if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
#                     #if dis[i].cpu() <dis_min-radius.cpu():
#                     if logits.data.max(1)[0][i].cpu()<logits_min[int(predictions[i].cpu())]-loss_r.cpu():#todo loss_r==1.2909
#                         predictions[i] = unknow#todo .max(1)[0]表示取的是每一行的最大值,[i]表示是第几个最大值
#                 correct_u += (predictions == labels.data).sum()
#
#                 _pred_u.append(logits.data.cpu().numpy())#todo 测试未知类的数据进行保存
#
#         tar = np.array([])
#         pre = np.array([])
#         for batch_idx, (data, labels) in enumerate(full_testloader):
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()
#
#             with torch.set_grad_enabled(False):
#
#                 data = data.squeeze(1).permute(0, 3, 1, 2)
#                 x, y = net(data, True)
#                 # x, y = net(data, return_feature=True)
#                 logits, dis, radius = criterion(x, y)
#                 predictions = logits.data.max(1)[1]
#
#                 total_all += labels.size(0)
#
#                 for i in range(predictions.shape[0]):
#                     #print('unknown:' + str(dis[i]))
#                     # if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
#                     # if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
#                     # if dis[i].cpu() <dis_min-radius.cpu():
#                     #if logits.data.max(1)[0][i].cpu()<logits_min[int(predictions[i].cpu())]-loss_r.cpu():
#                     if logits.data.max(1)[0][i].cpu() < logits_min[int(predictions[i].cpu())] - loss_r.cpu():
#                         predictions[i] = unknow
#                 correct_all += (predictions == labels.data).sum()
#
#                 tar = np.append(tar, labels.data.cpu().numpy())#todo 所有的
#                 pre = np.append(pre, predictions.data.cpu().numpy())
#
#                 #_pred_u.append(logits.data.cpu().numpy())
#
#     # Accuracy
#     acc = float(correct) * 100. / float(total)
#     print('Acc_k: {:.5f}'.format(acc))
# #add todo 保证被除数不为0
#     acc_u = 0
#     if float(total_u)!=0:
#         acc_u = float(correct_u) * 100. / float(total_u)
#     print('Acc_u: {:.5f}'.format(acc_u))
#
#     acc_all = float(correct_all) * 100. / float(total_all)
#     print('Acc_all: {:.5f}'.format(acc_all))
#
#     _pred_k = np.concatenate(_pred_k, 0)
#     _pred_u = np.concatenate(_pred_u, 0)
#     _labels = np.concatenate(_labels, 0)
#
#
#     # Out-of-Distribution detction evaluation
#     x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)#todo 获取每一个测试点对于概率最大的值
#     results = evaluation.metric_ood(x1, x2)['Bas']#todo 计算tnr,auroc,dtacc,auout,auin
#
#     # OSCR
#     _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)
#
#     results['ACC'] = acc
#     results['OSCR'] = _oscr_socre * 100.
#
#     return results,pre,tar


