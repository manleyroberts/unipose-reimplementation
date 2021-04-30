import torch

def calc_dists(preds, target):
  dists = torch.linalg.norm(preds - target)
  return dists

def threshold_acc(dists, threshold):
  return (dists < threshold).sum()

def evaluation(model_output, ground_truth, kpt):
  head_dist = calc_dists(ground_truth[0, 8], ground_truth[0,9]) # neck to head
  torso_dist = calc_dists(ground_truth[0, 6], ground_truth[0,7]) # waist to torso

  pckh_threshold = .5 * head_dist
  pck_threshold = .2 * torso_dist
  pckh = 0
  pck = 0
  count = 0

  for j in range(16): #each joint
    if(not (kpt[j, 0] < 0 or kpt[j, 1] < 0)):
      dist_heatmap = calc_dists(model_output[0, j], ground_truth[0, j])
      pckh += threshold_acc(dist_heatmap, pckh_threshold)
      pck += threshold_acc(dist_heatmap, pck_threshold)

      count += 1

  # test distance accuracy function
  pck_avg = pck/count
  pckh_avg = pckh/count

  return pck_avg, pckh_avg