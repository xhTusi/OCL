import torch
import torch.nn.functional as F
from requests import options
from torch.autograd import Variable
from utils import AverageMeter
import numpy as np


def CACLoss(distances, labels):
    '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
    true = torch.gather(distances, 1, labels.view(-1, 1)).view(-1)
    non_gt = torch.Tensor([[i for i in range(8) if labels[x] != i] for x in range(21)]).long().cuda()
    others = torch.gather(distances, 1, non_gt)

    anchor = torch.mean(true)

    tuplet = torch.exp(-others + true.unsqueeze(1))
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

    total = 0.1 * anchor + tuplet

    return total, anchor, tuplet

# Training
def train(net,  optimizer, trainloader, epoch=None, **options):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correctDist = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if options['use_gpu']:
            inputs, labels = inputs.cuda(), labels.cuda()#todo [21,1,7,7,200],[21,1]

        #
        # convert from original dataset label to known class label
        # targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)

        optimizer.zero_grad()
        # todo
        outputs = net(inputs)#todo todo [21,8],[21,8]
        print(outputs)
        labels = labels.to(torch.long)
        print(labels)
        cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], labels)  # outDistance

        # if args.tensorboard and batch_idx % 3 == 0:
        #     writer.add_scalar('train/CAC_Loss', cacLoss.item(), batch_idx + epoch * len(trainloader))
        #     writer.add_scalar('train/anchor_Loss', anchorLoss.item(), batch_idx + epoch * len(trainloader))
        #     writer.add_scalar('train/tuplet_Loss', tupletLoss.item(), batch_idx + epoch * len(trainloader))

        cacLoss.backward()

        optimizer.step()

        train_loss += cacLoss.item()

        _, predicted = outputs[1].min(1)

        total += labels.size(0)
        correctDist += predicted.eq(labels).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correctDist / total, correctDist, total))
#todo
    return predicted,correctDist
#
# def train(net, criterion, optimizer, trainloader, epoch=None, **options):
#     net.train()
#     losses = AverageMeter()
#
#     torch.cuda.empty_cache()
#
#     label_list=np.array([])
#
#     loss_all = 0
#     pre = np.array([])
#
#     pre_dis = np.array([])
#     for batch_idx, (data, labels) in enumerate(trainloader):
#         if options['use_gpu']:
#             data, labels = data.cuda(), labels.cuda()
#
#         with torch.set_grad_enabled(True):
#             optimizer.zero_grad()
#             data=data.squeeze(1).permute(0,3,1,2)
#
#             x, y = net(data, True)
# #todo add
#             # cacLoss, anchorLoss, tupletLoss = loss.CACLoss(outputs[1], labels)  # outDistance
#             logits, loss, loss_r,radius,dist_u = criterion(x, y, labels)
#
#             logits_max = logits.data.max(1)[0]
#             pre = np.append(pre, logits_max.data.cpu().numpy())
#             pre_dis= np.append(pre_dis, dist_u.data.cpu().numpy())
#             label_list= np.append(label_list, labels.data.cpu().numpy())
#
#             print(str(loss_r))
#             print(radius)
#
#             loss.backward()
#             optimizer.step()
#
#         losses.update(loss.item(), labels.size(0))
#
#         if (batch_idx+1) % options['print_freq'] == 0:
#             print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
#                   .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
#
#         loss_all += losses.avg
#     class_min=0
#     class_list_min=[]
#     #class_list_max=[]
#     for i in range(int(label_list.max())+1):
#         if pre[label_list==i].min()>class_min:
#             class_min=pre[label_list==i].min()
#         class_list_min.append(pre[label_list==i].min())
#         #class_list_max.append(pre[label_list==i].max())
#
#     #return loss_all, pre.min(), pre_dis.min(), loss_r
#     #return loss_all, class_min, pre_dis.min(), loss_r
#     return loss_all,class_list_min, pre_dis.min(), loss_r#todo _, logits_min, dis_min, loss_r
#
# def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG,
#         trainloader, epoch=None, **options):
#     print('train with confusing samples')
#     losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()
#
#     net.train()
#     netD.train()
#     netG.train()
#
#     torch.cuda.empty_cache()
#
#     loss_all, real_label, fake_label = 0, 1, 0
#     for batch_idx, (data, labels) in enumerate(trainloader):
#         gan_target = torch.FloatTensor(labels.size()).fill_(0)
#         if options['use_gpu']:
#             data = data.cuda(non_blocking=True)
#             labels = labels.cuda(non_blocking=True)
#             gan_target = gan_target.cuda()
#
#         data, labels = Variable(data), Variable(labels)
#
#         noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
#         if options['use_gpu']:
#             noise = noise.cuda()
#         noise = Variable(noise)
#         fake = netG(noise)
#
#         ###########################
#         # (1) Update D network    #
#         ###########################
#         # train with real
#         gan_target.fill_(real_label)
#         targetv = Variable(gan_target)
#         optimizerD.zero_grad()
#         output = netD(data)
#         errD_real = criterionD(output, targetv)
#         errD_real.backward()
#
#         # train with fake
#         targetv = Variable(gan_target.fill_(fake_label))
#         output = netD(fake.detach())
#         errD_fake = criterionD(output, targetv)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         optimizerD.step()
#
#         ###########################
#         # (2) Update G network    #
#         ###########################
#         optimizerG.zero_grad()
#         # Original GAN loss
#         targetv = Variable(gan_target.fill_(real_label))
#         output = netD(fake)
#         errG = criterionD(output, targetv)
#
#         # minimize the true distribution
#         x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
#         errG_F = criterion.fake_loss(x).mean()
#         generator_loss = errG + options['beta'] * errG_F
#         generator_loss.backward()
#         optimizerG.step()
#
#         lossesG.update(generator_loss.item(), labels.size(0))
#         lossesD.update(errD.item(), labels.size(0))
#
#
#         ###########################
#         # (3) Update classifier   #
#         ###########################
#         # cross entropy loss
#         optimizer.zero_grad()
#         x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
#         _, loss = criterion(x, y, labels)
#
#         # KL divergence
#         noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
#         if options['use_gpu']:
#             noise = noise.cuda()
#         noise = Variable(noise)
#         fake = netG(noise)
#         x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
#         F_loss_fake = criterion.fake_loss(x).mean()
#         total_loss = loss + options['beta'] * F_loss_fake
#         total_loss.backward()
#         optimizer.step()
#
#         losses.update(total_loss.item(), labels.size(0))
#
#         if (batch_idx+1) % options['print_freq'] == 0:
#             print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
#             .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
#
#         loss_all += losses.avg
#
#     return loss_all
