from sre_parse import FLAGS
import tensorflow as tf
import tf_slim as slim
#import argparse
import torch

#parser = argparse.ArgumentParser(description="Argument parser")
#parser.add_argument("--some_flag", type=str, help="Description of the flag")
#FLAGS = parser.parse_args()

class MultiLossLayer(torch.nn.Module):
    def __init__(self, num_losses, seed):
        super(MultiLossLayer, self).__init__()
        # Initialize sigma_sq for each loss as a trainable parameter
        torch.manual_seed(seed)
        self.s = torch.nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_list):
        total_loss = 0
        for i, individual_loss in enumerate(loss_list):
            sigma_sq = torch.exp(self.s[i])
            factor = 1 / (2 * sigma_sq)
            weighted_loss = factor * individual_loss + self.s[i] / 2
            total_loss += weighted_loss
        return total_loss

def get_loss(logits, ground_truths):
  multi_loss_class = None
  loss_list = []
  if FLAGS.use_label_type:
    if FLAGS.need_resize:
      label_type = tf.image.resize_images(ground_truths[0], [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    else:
      label_type = ground_truths[0]
    loss_list.append(loss(logits[0], label_type, type='cross_entropy'))
  if FLAGS.use_label_inst:
    xy_gt = tf.slice(ground_truths[1], [0, 0, 0, 0], [-1, FLAGS.output_height, FLAGS.output_width, 2])    # to get x GT and y GT
    mask = tf.slice(ground_truths[1], [0, 0, 0, 2], [-1, FLAGS.output_height, FLAGS.output_width, 1])  # to get mask from GT
    mask = tf.concat([mask, mask], 3)  # to get mask for x and for y
    if FLAGS.need_resize:
      xy_gt = tf.image.resize_images(xy_gt, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      mask = tf.image.resize_images(mask, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    loss_list.append(l1_masked_loss(tf.multiply(logits[1], mask), xy_gt, mask))
  if FLAGS.use_label_disp:
    if FLAGS.need_resize:
      gt_sized = tf.image.resize_images(ground_truths[2], [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      gt_sized = gt_sized[:, :, :, 0]
      mask = gt_sized[:, :, :, 1]
    else:
      gt_sized = tf.expand_dims(ground_truths[2][:, :, :, 0], axis=-1)
      mask = tf.expand_dims(ground_truths[2][:, :, :, 1], axis=-1)
    loss_list.append(l1_masked_loss(tf.multiply(logits[2], mask), tf.multiply(gt_sized, mask), mask))
  if FLAGS.use_multi_loss:
    loss_op, multi_loss_class = calc_multi_loss(loss_list)
  else:
    loss_op = loss_list[0]
    for i in range(1, len(loss_list)):
      loss_op = tf.add(loss_op, loss_list[i])
  return loss_op, loss_list, multi_loss_class

def calc_multi_loss(loss_list):
  multi_loss_layer = MultiLossLayer(loss_list)
  return multi_loss_layer.get_loss(), multi_loss_layer

def l1_masked_loss(logits, gt, mask):
  valus_diff = tf.abs(tf.subtract(logits, gt))
  L1_loss = tf.divide(tf.reduce_sum(valus_diff), tf.add(tf.reduce_sum(mask[:, :, :, 0]), 0.0001))
  return L1_loss

def loss(logits, labels, type='cross_entropy'):
  if type == 'cross_entropy':
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy, name='loss')
  if type == 'l2':
    return tf.nn.l2_loss(tf.subtract(logits, labels))
  if type == 'l1':
    return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(logits, labels)), axis=-1))



