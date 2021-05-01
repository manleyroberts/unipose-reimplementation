import torch
import numpy as np

# For example usage, check pck_pckh.ipynb

def calc_dists(preds, target):
  # x1,y1 = preds
  # x2, y2 = target

  # y = np.power((y2 - y1),2)
  # x = np.power((x2 - x1),2)
  # dist = np.sqrt(y + x)

  pred_arr = np.array(preds)
  target_arr = np.array(target)
  dist = np.linalg.norm(pred_arr - target_arr)

  return dist

def threshold_acc(dists, threshold):
  return (dists < threshold).sum()

def evaluation(model_output, ground_truth, kpt):

  # Threshold Calculation
  head1_cpu = ground_truth[0,8].cpu().detach().double().numpy()
  head1_argmax = np.argmax(head1_cpu)
  head1_unravel = np.unravel_index(head1_argmax, (46,46))

  head2_cpu = ground_truth[0,9].cpu().detach().double().numpy()
  head2_argmax = np.argmax(head2_cpu)
  head2_unravel = np.unravel_index(head2_argmax, (46,46))

  torso1_cpu = ground_truth[0,6].cpu().detach().double().numpy()
  torso1_argmax = np.argmax(torso1_cpu)
  torso1_unravel = np.unravel_index(torso1_argmax, (46,46))

  torso2_cpu = ground_truth[0,7].cpu().detach().double().numpy()
  torso2_argmax = np.argmax(torso2_cpu)
  torso2_unravel = np.unravel_index(torso2_argmax, (46,46))

  if ((torso1_unravel[0] < 1) or (torso1_unravel[1] < 1) or (torso2_unravel[0] < 1) or (torso2_unravel[1] < 1) or (head1_unravel[0] < 1) or (head1_unravel[1] < 1) or (head2_unravel[0] < 1) or (head2_unravel[1] < 1)):
    return -1, -1

  head_dist = calc_dists(head1_unravel, head2_unravel)
  torso_dist = calc_dists(torso1_unravel, torso2_unravel)

  # head_dist = np.linalg.norm(head1_unravel - head2_cpu) # neck to head
  # torso_dist = np.linalg.norm(torso1_cpu - torso2_cpu) # waist to torso

  pckh_threshold = .5 * head_dist
  pck_threshold = .2 * torso_dist

  # print("THRESHOLDS")
  # print(pckh_threshold)
  # print(pck_threshold)
  
  pckh = 0
  pck = 0
  count = 0

  for j in range(16): #each joint
    if(not (kpt[j, 0] < 0 or kpt[j, 1] < 0)):
      model_cpu = model_output[0,j].cpu().detach().double().numpy()
      model_argmax = np.argmax(model_cpu)
      model_unravel = np.unravel_index(model_argmax,(46,46))

      # print("Model Unravel " + str(j))
      # print(model_unravel)

      gt_cpu = ground_truth[0,j].cpu().detach().double().numpy()
      gt_argmax = np.argmax(gt_cpu)
      gt_unravel = np.unravel_index(gt_argmax,(46,46))

      # print("Ground Truth Unravel " + str(j))
      # print(gt_unravel)

      dist_heatmap = calc_dists(model_unravel, gt_unravel)
      # print("Dist Heatmap")
      # print(dist_heatmap)
      pckh += threshold_acc(dist_heatmap, pckh_threshold)
      pck += threshold_acc(dist_heatmap, pck_threshold)

      count += 1

  # test distance accuracy function
  pck_avg = pck/count
  pckh_avg = pckh/count

  return pck_avg, pckh_avg