from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from src.data.SupContrast.losses import SupConLoss
from src.data.loss_handler import MultiLossLayer
from src.training.invariables import DEV


def init_loss_functions(seed):
    global BINARY_CORSS_ENTROPY_LOSS, CORSS_ENTROPY_LOSS, CONTRASTIVE_LOSS_FUNCTION, MULTI_LOSS_LAYER
    BINARY_CORSS_ENTROPY_LOSS = BCEWithLogitsLoss().to(DEV)
    CORSS_ENTROPY_LOSS = CrossEntropyLoss().to(DEV)
    CONTRASTIVE_LOSS_FUNCTION = SupConLoss(div=DEV)
    MULTI_LOSS_LAYER = MultiLossLayer(num_losses=2, seed=seed).to(DEV)



def vfd_loss_function(classification, normalized_projection_features, label):

    classification_loss = CORSS_ENTROPY_LOSS(classification, label)
    contrastive_loss = CONTRASTIVE_LOSS_FUNCTION(normalized_projection_features, label)


    # Combine losses
    loss = MULTI_LOSS_LAYER([contrastive_loss, classification_loss])


    return loss


# Set uo loss function for all other models:
def non_vfd_loss_function(classification, normalized_projection_features, label):

    loss = CORSS_ENTROPY_LOSS(classification, label)
    return loss



def vfd_loss_function_binary(classification, normalized_projection_features, label):

    classification_loss = BINARY_CORSS_ENTROPY_LOSS(classification, label)
    contrastive_loss = CONTRASTIVE_LOSS_FUNCTION(normalized_projection_features, label)


    # Combine losses
    loss = MULTI_LOSS_LAYER([contrastive_loss, classification_loss])


    return loss


# Set uo loss function for all other models:
def non_vfd_loss_function_binary(classification, normalized_projection_features, label):

    loss = BINARY_CORSS_ENTROPY_LOSS(classification, label)
    return loss
