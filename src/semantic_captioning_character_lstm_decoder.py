from typing import List, Tuple, Optional

from spacy_universal_sentence_encoder import load_model
from numpy import array, log2
from numpy.random import choice
from keras import Model
from keras.layers import Input, Dropout, Dense, LSTM, Embedding, add
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class SemanticCaptioningCharacterLSTMDecoder:
    """
    generates a relevant question
    when given the semantics of a sentence
    using a custom-trained character-based LSTM 
    and a pre-trained sentence encoder
    """

    def __init__(self,recursion_depth:int,path_to_model_weights:Optional[str]=None) -> None:
        self.recursion_depth = recursion_depth
        self.semantic_encoder = load_model('en_use_lg')
        a = ord('a')
        z = ord('z')
        is_valid_character = lambda character: character.isspace() or a <= ord(character) <= z
        self.clean_string = lambda text:''.join(filter(is_valid_character,text.lower().strip()))
        self.start_token='|'
        self.stop_token='?'
        character_set = [' '] + list(map(chr,range(a,z+1))) + [self.start_token,self.stop_token]
        self.index_character_mapping = dict(enumerate(character_set))
        self.character_index_mapping = {
            character:index for index,character in self.index_character_mapping.items()
        }
        self.number_of_characters = len(character_set)
        self.model = self.build_character_based_LSTM(
            semantic_vector_length=len(self._get_semantic_vector("")),
            character_vector_length=self.number_of_characters,
            character_sequence_length=self.recursion_depth,
            hidden_layer_length=256,
            dropout_rate=.5,
            activation="relu",
            loss = "categorical_crossentropy",
            optimisation= "adam",
            weights=path_to_model_weights,
        )

    def train(
        self, questions:List[str], 
        batch_size:int, epochs:int, 
        save_to_file_path:str="char_lstm_weights.hdf5",
        question_contexts:Optional[List[str]]=None
    ) -> None:
        """
        if only questions are provided (e.g. from the Quora Dataset),
        the model is trained only on the unlabelled questions 
        (attempting to do a type of Unsupervised Learning)
        in a fashion similar to Image Captioning (Image->Text)
        but using Semantics instead - Semantic Captioning (semantics->text)

        if question_contexts are provided too (the prompt that triggers each question)
        then the model similarly does Semantic Captioning (Contexts->Text)
        but uses the semantics of the question_contexts instead of the Questions
        (since the question_contexts are mapped to the questions, like labels, 
        the model is now being trained via supervised learning)
        """
        steps_per_epoch = len(questions)//batch_size
        for epoch in range(epochs):
            example_sentence = str(choice(questions)) if question_contexts is None else str(choice(question_contexts))
            print(f"epoch {epoch}: {example_sentence}\n\tgreedy: {self.generate(example_sentence)}\n\ttop-p=.8,temperature=1.2 {self.generate(example_sentence,top_p=.8,temperature=1.2)}\n\ttemperature = 1.2: {self.generate(example_sentence,temperature=1.2)}\n\ttemperature = .5: {self.generate(example_sentence,temperature=.5)}")
            self.model.fit(
                self._get_training_data(questions, batch_size, question_contexts), 
                verbose=True, epochs=1, steps_per_epoch=steps_per_epoch, 
            )
            self.model.save_weights(save_to_file_path)

    def _get_training_data(self,questions:List[str],batch_size:int,contexts:Optional[List[str]]) -> Tuple[Tuple[array,array],array]:
        iteration_count,X_meaning, X_characters, Y_characters = 0,[],[],[]
        while True:
            for question_number,question in enumerate(questions):
                iteration_count+=1

                semantics = question if contexts is None else contexts[question_number]
                semantic_vector = self._get_semantic_vector(semantics)
                processed_question = self._preprocess_characters(question)
                character_indexes = self._convert_characters_to_index(processed_question)
                for index in range(1, len(character_indexes)):
                    contextual_character_indexes = character_indexes[:index]
                    next_character_index = character_indexes[index]                    
                    X_meaning.append(semantic_vector)
                    X_characters.append(self._pad(indexes=contextual_character_indexes))
                    Y_characters.append(self._one_hot_encode_output(next_character_index))

                if iteration_count==batch_size:
                    yield ([array(X_meaning), array(X_characters)],array(Y_characters))
                    iteration_count,X_meaning, X_characters, Y_characters = 0,[],[],[]

    def generate(self, sentence:str, temperature:Optional[float]=None, top_p:Optional[float]=None) -> str:
        """
        given a conversation
        a question is asked about it
        (if no temperature specified
        use greedy decoding by default)
        """
        conditional_vector = array([self._get_semantic_vector(sentence)])
        generated_question = self.start_token
        for _ in range(self.recursion_depth):
            generated_question += self._predict_next_character(
                meaning_vector=conditional_vector,
                contextual_characters=generated_question,
                temperature= temperature,
                top_percentage=top_p
            )
            if self.stop_token in generated_question:
                break
        return generated_question
  
    def _predict_next_character(self, meaning_vector:array,contextual_characters:str,temperature:Optional[float],top_percentage:Optional[float]) -> str:
        """
        pick most confident prediction
        this is local optimal but will not arrive at global optimal
        (use beam search for that)
        but optimal is not the most realistic decoding strategy
        better to pick a more varied range to add novelty (e.g. p-decoding)
        """
        characters_vector = array([self._pad(self._convert_characters_to_index(contextual_characters))])
        output_vector = self.model.predict((meaning_vector,characters_vector),verbose=False)
        predicted_index = self._nucleus_decode(output_vector,top_percentage,temperature) if top_percentage else self._temperature_decode(output_vector,temperature) if temperature else self._greedy_decode(output_vector)
        return self.index_character_mapping.get(predicted_index)


    def _convert_characters_to_index(self, characters:str) -> List[int]:
        return list(map(self.character_index_mapping.get,characters))

    def _pad(self, indexes:List[int]) -> array:
        return pad_sequences([indexes], maxlen=self.recursion_depth)[0]

    def _get_semantic_vector(self, text:str) -> array:
        """
        currently uses spacy's small pretrained wordvectors
        but better semantic representations of sentences can be achieved 
        using pretrained language models (e.g. universal sentence encoder)
        """
        if not any(text): text = self.start_token
        return self.semantic_encoder(text).vector

    def _one_hot_encode_output(self, index:int) -> array:
        return to_categorical([index], num_classes=self.number_of_characters)[0]

    def _preprocess_characters(self, text:str) -> str:
        return f"{self.start_token} {self.clean_string(text)} {self.stop_token}"

    @staticmethod
    def _nucleus_decode(predicted_vector:array, top_p:float, temperature:Optional[float]) -> int:
        probabilities = predicted_vector[0]
        probabilities /= probabilities.sum()
        ranked_indexes = sorted(range(probabilities.size), key=lambda index:probabilities[index],reverse=True)
        probabilities.sort()
        ranked_probabilities = probabilities[::-1]
        for index in range(ranked_probabilities.size):
            if ranked_probabilities[:index].sum() >= top_p:
                nuclues_probabilities = ranked_probabilities[:index]
                nuclues_indexes = ranked_indexes[:index]
                break
        if temperature: nuclues_probabilities *= temperature
        nuclues_probabilities /= nuclues_probabilities.sum()
        return choice(nuclues_indexes,1,p=nuclues_probabilities)[0]
    
    @staticmethod
    def _temperature_decode(predicted_vector:array,temperature:float) -> int:
        probabilities = predicted_vector[0]
        probabilities *= temperature
        probabilities /= probabilities.sum()
        return choice(range(probabilities.size),1,p=probabilities)[0]

    @staticmethod
    def _greedy_decode(predicted_vector:array) -> int:
        return predicted_vector[0].argmax()

    @staticmethod
    def build_character_based_LSTM(
        semantic_vector_length:int,
        character_vector_length:int,
        character_sequence_length:int,
        hidden_layer_length:int,
        dropout_rate:float,
        optimisation:str,
        activation:str,
        weights:Optional[str],
        loss:str,
    ) -> Model:

        meaning_layer1 = Input(shape=(semantic_vector_length,))
        meaning_dropout1 = Dropout(dropout_rate)(meaning_layer1)
        meaning_layer2 = Dense(hidden_layer_length, activation=activation)(meaning_dropout1)
        characters_layer1 = Input(shape=(character_sequence_length,))
        characters_layer2 = Embedding(character_vector_length, character_vector_length, mask_zero=True)(characters_layer1)
        characters_dropout2 = Dropout(dropout_rate)(characters_layer2)
        characters_layer3 = LSTM(hidden_layer_length,return_sequences=True)(characters_dropout2)
        characters_dropout3 = Dropout(dropout_rate)(characters_layer3)
        characters_layer4 = LSTM(hidden_layer_length,return_sequences=True)(characters_dropout3)
        characters_dropout4 = Dropout(dropout_rate)(characters_layer4)
        characters_layer5 = LSTM(hidden_layer_length)(characters_dropout4)
        layer3 = add([meaning_layer2, characters_layer5])
        layer4 = Dense(hidden_layer_length, activation=activation)(layer3)
        layer5 = Dense(character_vector_length, activation='softmax')(layer4)
        model = Model(
            inputs=[meaning_layer1, characters_layer1], 
            outputs=layer5
        )
        if weights is not None: model.load_weights(weights)
        model.compile(loss=loss, optimizer=optimisation)
        return model 