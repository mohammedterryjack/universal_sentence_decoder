from utils.dailydialog_loader import load_daily_dialog
from src.semantic_captioning_character_lstm_decoder import SemanticCaptioningCharacterLSTMDecoder

contexts,questions = zip(*load_daily_dialog("data/daily_dialog"))
decoder = SemanticCaptioningCharacterLSTMDecoder(recursion_depth=100)
decoder.train(
    question_contexts=contexts,
    questions=questions,
    batch_size=5,
    epochs=100,
    save_to_file_path="pretrained_models/intent_captioning_daily_dialog.hdf5"
) 