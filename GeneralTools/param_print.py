# =========================================================================
# This code built for the Competition of iNaturalist 2019 at FGVC6 
# Part of MCEN90048 project
# This file contains functions to print hyper-parameters
# Written by Zhuo Li
# The Univerivsity of Melbourne
# zhuol7@student.unimelb.edu.au
# =========================================================================

from GeneralTools.misc_fun import FLAGS

def param_print(version = FLAGS.VERSION,
                model = FLAGS.MODEL,
                batch_size_tr = FLAGS.BATCH_SIZE_TRAIN,
                batch_size_val = FLAGS.BATCH_SIZE_VAL,
                buffer_size_tr = FLAGS.BUFFER_SIZE_TRAIN,
                buffer_size_val=FLAGS.BUFFER_SIZE_VAL,
                target_size = FLAGS.TARGET_SIZE,
                epoch = FLAGS.EPOCH,
                optimizer = FLAGS.OPTIMIZER,
                lr = FLAGS.LR):

    print('==================================================')
    print('Vision:            {}'.format(version))
    print('Pre-train model:   {}'.format(model))
    print('==================================================')
    print('Batch size:        ' + '{}(train), '.format(batch_size_tr) + '{}(validation)'.format(batch_size_val))
    print('Buffer size:       ' + '{}(train), '.format(buffer_size_tr) + '{}(validation)'.format(buffer_size_val))
    print('Target size:       ' + str(target_size)+ '*' + str(target_size))
    print('Total epoch:       {}'.format(epoch))
    print('Optimizer:         {}'.format(optimizer))
    print('Learning Rate:     {}'.format(lr))
    print('==================================================')

    return