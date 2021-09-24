from src.semantic_captioning_character_lstm_decoder import SemanticCaptioningCharacterLSTMDecoder

decoder = SemanticCaptioningCharacterLSTMDecoder(
    recursion_depth=100,
    path_to_model_weights="pretrained_models/daily_dialog_epoch31.hdf5"
)
while True:
    text = input("> ")
    question = decoder.generate(text,1.5,.9)
    print(question)