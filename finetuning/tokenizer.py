# tokenizer.py

from typing import List, Union, Dict, Optional
import torch
from commons import intersperse

class VITSTokenizer:
    def __init__(self):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        
        self.symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        
        self.vocab = {s: i for i, s in enumerate(self.symbols)}
        
        if 'U' not in self.vocab:
            self.vocab['U'] = len(self.vocab)
            
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab[_pad]
        self.unk_token_id = self.vocab['U']

    def encode(self, text: str) -> List[int]:
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text]
        
        tokens = intersperse(tokens, 0)
        
        return tokens
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> dict:
        if isinstance(text, str):
            input_ids = self.encode(text)
        else:
            input_ids = [self.encode(t) for t in text]
            
        if isinstance(text, str):
            attention_mask = [1] * len(input_ids)
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask)
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(ids) for ids in input_ids]
            }

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, 'U')
            if skip_special and token == "$":
                continue
            tokens.append(token)
        return "".join(tokens)

    def __len__(self):
        return len(self.vocab)

class PLBERTTokenizer:
    def __init__(self):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        
        self.symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        
        self.vocab = {s: i for i, s in enumerate(self.symbols)}
        
        if 'U' not in self.vocab:
            self.vocab['U'] = len(self.vocab)
            
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab[_pad]
        self.unk_token_id = self.vocab['U']

    def encode(self, text: str) -> List[int]:
        tokens = []
        for c in text:
            if c in self.vocab:
                tokens.append(self.vocab[c])
            else:
                tokens.append(self.unk_token_id)
        return tokens
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> dict:
        if isinstance(text, str):
            input_ids = self.encode(text)
        else:
            input_ids = [self.encode(t) for t in text]
            
        if isinstance(text, str):
            attention_mask = [1] * len(input_ids)
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask)
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(ids) for ids in input_ids]
            }

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, 'U')
            if skip_special and token == "$":
                continue
            tokens.append(token)
        return "".join(tokens)

    def __len__(self):
        return len(self.vocab)

class XPhoneBERTTokenizer:

    SYMBOLS = '<s> <pad> </s> <unk> ▁ n a t i s ə d ɪ k e l o ɛ m j r p u ɾ ɔ b v z ʊ , f ɑ ɐ . ɯ ŋ ɹ w ʃ ɫ ʁ ˥˩ h ɡ ˈɛ ɕ iː ˧˥ ˈɪ aː ð ˈæ æ ˈe x eː ɨ ˥˥ ˈa g ʒ ʔ ʂ ɐ̯ ˈt ˈə t͡s ɝ ˨˩˦ : y ˈs ˈð ɕʰ ˈi ( ) oː ˈk ˈʔ ˈf uː ɲ lʲ ˈɒ ʈ ˈm t͡ʃ nʲ ɴ ʋ ˈb ˈv - tʲ ˈh tʰ ˈiː ˈuː ˈd ˈp ɤ ɑː ˧˧ θ ç ɒ ˈɑ ˈo ɑ̃ ɣ vʲ ˈw ʌ ˈɔː rʲ ˈn ɦ ˈɑː d͡ʒ œ sʲ " ɛː t͡ɕ t̪ ɪ̯ ʐ ˦˥ ˈu β mʲ ˈʌ ˈʃ ˌa ˈɫ ɔ̯ ˧˨ ʏ ˨ˀ˩ ɚ c õ ˈz ø kʰ ũ̯ ʎ i̯ ˈɜː ˌɪ ɔ̃ a̠ ɒː ẽ dʲ kʲ ɸ d̪ n̪ u̯ ɥ ˈl ˈj ˈɔ ˧˩˨ ɭ æ̃ sʰ o̝ q ˈɡ ˈɾ ɐ̃ ɤ̆ ˈɹ ŋ͡m ˌe pʲ ʂʰ ˌɛ a̝ ʑ pʰ l̪ d͡z ă ʌ̹ ˧ ˦˨ ʕ ˨˩˨ ˌt d͡ʑ ˈɯ ɛ̃ ˩˧ yː ˈg ɟ ˨˦ ˌi ; äː o̞ bʲ ˌo ˈʊ nː d̥ ĩ̯ ʊ̯ s̻ A zʲ ɾ̪ ˌk s̺ øː s̪ ˌæ ɡʲ t͡ʂ ɕː j̃ ˈɐ ɵ e̝ ɔː E ˨˩ w̃ ʀ ˌs ħ fʲ “ n̩ t͡sʰ kː tː ⁽ʲ ˥ ʋʲ ⁾ œ̃ k͡p \' ũ O ˈeː ˌn e̞ ˌu ˌd t͡ʃʰ ɖ I % ˌiː ĭ ã ˌm ê っ ĩ χ t͡sʲ â ɰᵝ ɳ ǒ ɛ̝ ɡ̊ ˌɒ ô ˌə ˈaː ˨ ˌv ˌʃ ˌf / ˈθ ` ʝ ˈy N ˌp pː ʉ t̻͡͡s b̥ ˌb ɔ́ ” tˤ ɻ ǎ k̚ ˌl ˈɨ nʲː sˤ ʧ r̩ ˈoː t͡ɕʰ ɽ ˌh ˧ˀ˥ t̪ʰ ˞ t͡s̻ tʼ tˢ ě ˌɫ î s̠ ˌj r̝̊ ǎː y̯ ɰ ʈ͡͡ʂ ˌuː sː áː kʼ bʱ ʈː d̪ʱ ˌɾ t̚ ˌʔ ǐ ˌz ɢ ˌɑ ˌw 0 ˌɹ ˈa͡ɪ U ûː } xʲ 2 dˤ ˌg a͡ɪ nˀ ɭʲ ˌɡ t̪ː r̝ ə̃ óː éː ˌɔ ˌɑː æː ˌɐ t͡sʼ ìː ɾʲ ôː ˈyː ɯː âː ˌɔː ˩˩˦ ɾˠ û ø̞ àː ˈr íː ɛ́ ĕ lː pʼ í ɨ̯ t͈ ˌʌ ʃʲ ǔ ˈç eːˀ & ä̃ː C nˠ æːˀ ] sˠ ěː p̚ ɜː ˈɝ ˌʊ á [ ˈβ m̩ ˈœ ... èː d͡ʐ ˌaː bː ! * S l̩ k͈ ʁ̥ ɨ̞ ˈj͡a ˈɑʲ à ẽː a͡ʊ a̯ ɟː ˈø ˈʁ ˈɵ lˀ t͡sː qʼ T ʍ ǐː ǒː ɬ r̥ ^ t̪ˠ n̪ˠ ? ɦʲ a̰ ð̩ wˀ ɫ̪ ɔˀ ðˤ ˈx ˈa͡ʊ 9 ʏ̯ ɗ ŏ ɱ ˈɣ úː êː îː ˌr ˌɨ ɓ t͡ɕ͈ t͡sʲː õː ˈʀ eˀ ɪː ɭː ʒʲ ʈʰ ʃː ɑˀ ʼa ɛˀ d̪ˠ o̯ t͡s̺ ʦ t̻͡͡sʲ ɐ̯ˀ dʲː ðˀ r̂ iːˀ é è t͡ɕː dː s͈ æˀ ʲˈj͡a mˀ e̯ oˀ mˠ ‘ mː F D Y jː ó ʊ̃ˑ R ˌɜː l͡l t͡ʃʲ tʲː ɡʱ ˌy aʲ ǔː ŋ̩ ř ɪ̃ | ʄ rː l̪ˠ aˑ l̥ ʲ̩ t͡ʃː ɪ̀ lˠ ʼo l̪ː ŋˀ ɛ̌ː ˈe͡ɪ n̠ʲ # œː e͡ʊ ˈt͡s ɛ̯ > ɛ̃ː oːˀ d͡zʲ iˀ ʃ̩ bˠ ɤː ɒˀ ɸʷ ˌeː ˌθ jᵊ ˈʒ ì e͡ɪ ɑːˀ ˈɜ͡ɪ ɨː ò ɛ̀ n̥ ùː l̠ l̠ʲ ɠ ə́ jˀ ː͡ɪ fˠ ɐ́ ʊ̃ ʏː ḭ ɔ̀ː ĩː ˈuʲ ʲɪ ˌœ bʰ < cː ⁾ː ˈŋ ˈə͡ʊ o͡ɪ ʑː òː ɣʲ ɶ ˈɑ͡ʊ ɔ́ː p͈ zː ɔ̌ː ˈøː øˀ ˈo͡ɪ ɫː ʂː ɛ̀ː ˈɑ͡ɪ ṵ m̥ ˌoː t͡ʃʼ sˤˤ o̰ ʋː ˈæː ɛ̰ tˤˤ ˈʎ ə̀ pˠ aⁿ cʰ ɑ̃ː ˌɵ ˈɲ ˈu͡ə ˌɯ ˈæ͡i lʲː ˈi͡ə ɲː d͡ʒʱ X i͡ə yˀ { ù ɡː ɪ̰ ɕ͈ ɱ̩ ʼɔ ũː ɪ́ ˈʊː ɜ͡ɪ ɾᵊ ˈʝ ãˑ ɛ́ː r̂ː ʼɑ ɔ̀ ḛ ˈɛː t̩ ú ʼi ² t̪̚ sʲː ʼy kᶣ ɛ̃ˑ vʲː ɳː uːˀ uˀ ə͡ʊ kʷ ʲˈɔ ˌð ˈœː ɪ̯ˑ ʼɥ aᵐ vˠ ʋ̥ iᵑ M ~ ˈʧ d͡ʑː ˈɔ͡ɪ ʼɑ̃ ş ɒːˀ ɾː t͡ʃʲː ɫ̩ ðˤˤ ʊ̯ˑ ː͡ʊ ˈi͡u ʼɛ ˌe͡ɪ ɪ̯̃ ʼe ˌaʲ ʲ̩̩ ˈãː yːˀ fː u͡ə ˈe͡ʊ ʁː ɑ͡ɪ ä̌ː vː ɖʱ ˈoʲ ŋ̊ ˈɑ̃ː aᵑ ˈɛʲ ˌa͡ɪ ˌyː gː ɛːˀ ˈj͡aʲ ˈɑ́ː ˈʉː ⁴ œ̞ dˤˤ ˌä̃ː s̩ y͡ɛ uᵑ a̠ː ˈo͡ʊ K ʼu ɽʱ i͡u ³ mʲː ʒ̩ ˌø ʲˈj͡aʲ eⁿ pʲː i̝ ɑ͡ʊ iⁿ uⁿ P ˌo͡ɪ eᵑ d̪̚ ⁿɗ ˈy͡ɛ ʊ̀ uʲ ɭ̆ ý k̩ B øːˀ iᵐ ˌa͡ʊ ä̂ˑ ʲˌɛ W ɖː řː n̪ᵊ ɛʲ ɪ̌͡ə m̩ː ʼɔ̃ ˌi͡u ˈ ʼõ ʼaː oᵑ ˌäː ˌç ᵐɓ ɶː uᵐ ッ dʰ L æ͜ɑ š ˌʀ ˈC ʒʲː ˈi͡e ɑ́ː ɯ̟ᵝ ɛ̝ː ˈtʰ ʲɛ ɛ̠ ɸː ãː ʲˌɪ ɔːˀ ˈʋ ˌ ʷo hː ŋʲ o̞ː ˈʏ ð̠ˠˀ ˌʁ ⁻ G hʲ æ̯ e͡u t̪ʲ d͡ʒʲ ɔ̃ː wʲ n̪ː oⁿ wᵊ ʐː o͡ʊ kʲː ʼœ ɽː ˌæː rʲː f̩ ʼj æ͜ɑː ɪ̃ˑ u̝ ˈäː ˈẽ d͡ʒː β̞ eᵐ ʲˈɛ ʲ̩ʲ ɘ ¹ n̪ʲ ˈɘː ʲˈu ɞ d͡zː ð̞ ã̠ t͡ ə̯ ˈɵː ˈʂ mᵊ ʊ́ ˈt͡ʃ @ ˌʒ s̪ʲ i͡e ˌə͡ʊ H tʰː ˈɛ̃ː ɜ əː ɔ͡ɪ ʌʲ ɶˀ ɔ̰ ʼœ̃ ˈsʰ ʊː ʈ͡͡ʂʲ ɡʷ ˌĭ uˑ J ʲu ʲˌu ɫ̩ː d̪ʲ ʋʲː ˈʌ̃ o̩ ʃʰ æ̌ː xː oᵐ xʷ ɪ̂ˑ ɑ̯ rˠ ɨᵝ ˌɛː ^ː ʈ̚ ʋᵊ ɖʰ ˌɑ͡ʊ ʼə ᵑg ˌy͡ɛ ʼæ̃ ɡʰ ˈq ⁵ ˈĭ ˌæ͡i ˈt͡ɕ ɐ̃́ æ̞ ˌe͡ʊ ˈæ͡iː ǝ ˈĩː ˔ h̩ d̪ː r̩ː ðˠ y̌ː p̩ ˈe͡u ɡᶣ ɯ̃ ẽ̞ ˌt͡s ˈũː ˌŋ d͡ ˈʉ œˀ ʊ̯ˀ ɪ̂ ا ʼɛ̃ ˈe͡iː ɔˑ ɔ̩ ˌáː ð̝ ˈɔ̃ː ˌo͡ʊ kʰː aˀ ɭ̩ gʰ ˈr̩ b̩ ː́ ɨ̃ ˌʌʲ ɘː ˈa͡uː ʼw ˌɜ͡ɪ bʲː e̞ː س م wˠ ː ʉː ˌuʲ j̊ k̥ zʲː d͡ʐʲ ɪ̯ˀ χː çː ˌx ž aːˀ ɨᵝː e͜oː ˾ ˌi͡ə ˈä̃ː ʼã ˌɝ l̪ᵊ ˌɛʲ ʼẽ ˈʈ ˈχ ŝ dˠ ˈɪː ˈš I̯ e͜o ˈɕ bʱː ʣ æ̝ː ɖ̚ ˈəː ˈy͡i ˈɤ̆ æ͡i ˈă ˈɑ̃ ˈe͡i b̚ ʌ̃ ˌɑ͡ɪ t͡sʰː zˠ ɰ̃ a͜i i͜yː u̯ː ɐˀ ˈr̝ ɔ̂ː ˌɑ̃ː ˌøː aʼ ˌe͡u ˌu͡ə t͡S ɾ̪ː æ͡l d̩ ɲ̊ o̝̝ ẍ ː̃ ðˠˀ ˈø͡i ʐʲ äˑ ɟʰ ˌq i͜y d̚ ʼø ʼl :ː z̩ t͡ʃʰː ᵐv uˡ ˈc gʲ i͡uː ŋ̍ ˌʏ e̝ː œ̞ːˀ ˈa̠ː ⁿz ʊ̰ s̚ ɫʲ ɲ̟ ˌl̩ tˠ ü ˥˥˥ ˈŋ̩ fʲː ʂʲ ð̠ˠ ˈɤ fˀ ˈé ɛ̂ iˀː ɖᵊ ˈd͡ʒ v̩ jʲ s̻ː vʼ ʼʁ Iⁿ Z b̪ i͜u ˈˈɛ ɡ̥ ˌœː Eᵑ p̪ ʈʲ ʲɔ ʒː ˈa̠ ˈa͡u Eⁿ w̩ ˌi͡e œ̞ː j̩̩̩ Aᵑ ɜːˀ d͡ʑʱ ˥˩˦ ˈkʰ a͡u ˌʧ t̪͡s̪ Aⁿ ل t͡ʂʲ ˈáː ʋ̂ː õ̞ ʌː Eᵐ ʒ͡ʲ ˌš ä ˈɐ̯ ˈɛ͡ɪ ˈɔ́ ŋː a͡i œːˀ t͡s̪ʲ ˈd͡ʑ Ń u̩ ˈɐ̃ ˌi͡uː n̺ ɵː e̩ ˈl̩ ˈi̯ z̪ â͡l ˈɔ̃ ˈɔ̩ ɛ̝ˀ ʲˌɔ n̂ θː ǀ ̯ ˌɔ͡ɪ ɛ̂ː ˈɛ̝ː l̩ː kˠ ɪ̯̯ ˩˦ ʲˈɔʲ j̥ ˈħ ˈɶː j̩ ï n̠ ˈɾ̝ ˦ ʼoː ˈɶ ˈɯː l̩ʲ t͡g ˌɛ̃ː e͡uː ˈó ɾ̝ âˑ ˈɛ̃ ä̂ː ɐː ʰ n̪ˑ ˈɨː hˠ ŋ́ˑ ˌɤ ˈɟ l̺ mˑ ˌe͡uː ب ص ف ه iʼ k̩ʲ ˈɔʲ ˈɖ ˈE jˠ ˛ ɛ̆ ɪ̯̌ː θ̝ wː ŷː dʼ ˈS ˈɸ cʲ ɡ̩ ˈi͡uː l̂ ˈʁ̝ ˈž ˾˧ e͡i p̩ʲ l̪ʲ ɝː eʼ ˌʂ ˔˧ ø͡i ˈu̯ ˈæ̃ː ů ˈpʰ ɽᵊ ˈe͡uː t͡s̪ ˌa͡u Uᵑ y̆ t̪͡ʃ u̯ʲ ˈɳ ˪ ˌɲ j̩̩ ɑ̃ˀ ɔ̃ˑ "ː ʲ t̩̩̩̩ ˈʐ ʀ̥ ˈõː ʁ̝ ʎ̩ ˛˧˥ ĝ l̂ː ˈɤː ɦː æ̌ˑ ˧˥˥ ɛ̂ˑ vʱ ˈá ɮ i̯ˀ pʰː ʎː t̪͡s̪ʲ ˌɕ uʼ ʁ̝̊ ŕ̩ ɛʼ ˈæ͡ʊ n̂ː ʁ̩ æ̂ ɔ̝ ʼeː Oⁿ ˈɜ͡ʊ ʃ̩ʲ ˈʂʰ ʁʷ ˔˧˧ e͡iː ɖʱː k̩̩̩̩ ʲˈɛʲ ˈɛ̌ː ɯ̟̊ ʔ͡p ˩ æ͡ɪ a̩ vˀ ˈe̞ əᵊ y͜yː ɒ͜úˑ ɯ̟ᵝː ˌž a̝̝ cʼ b̥ˀ u̯ːˀ ƹ ˌø͡i ˈɪ̯ ˌC ðʲ ˌt͡ʃ sʼ ˈe̞ː ɡʲː ɔʼ ˌè ˈeːˀ ˌy͡i œ̆ ɖ̥ ˈɕʰ s̩ʲ ɾ̝̊ ˌé j̩̩̩̩ Aᵐ vʰ ˈl̩ː ˌc ˈÜː ˈɔ̂ː t̩̩̩ t͡sˠ θʲ ˈã ʼs ˌE ɛ͡ɪ ˈæ͡ɑ a͡iː i͜oː t̚ˠ Eː ˌæ̃ː ˌa̠ː t͡ʃ̩ ˨ˀ ɹ̝ Ɍ ɒ̃ mᵑ æ͡ʊ ʼɐ ˈQ i͜uː á̃ː ʃ̩̩ ˈɦ V ˈɸʷ ˌɯː œ̃ː ɾ̪ˠ ˕ d͡zʲː ɚː ˌS iˑ l̪ˑ t̪͡s ˈdʲ eʲ v̥ ˌɨː sˀ f̩̩̩̩ lˑ ɡːʲ á̃ ɹː k̩̩̩ ˌă ˈA tˢʰ çʲ ø̯ ŧ ɾˀ :ˀ ɐ̀ qʷ ʼʏ d͡ʂ hˀ vʷ ʈʰː Ü ɶːˀ ˧˧˧ ˧˩˨ˀ˩ m̂ n̩ʲ p̝ u̯ʲː ːt ŋˠ ł ɳːː ˈj͡u oˀː pʷ l̩̩ s̥ ɹʲ ˈi͡ɛ ˌe̞ bˠʲ d͡ʒʲː y̩ ɻː ˌʈ t̥ ɹˀ ăː ʼv ˈʁ̥ r̂ˑ ŋ̪ˠ ä̂ ʲˈo ˈs̺ ʒˠ ˌˌl l͡p oʻ ŋ̥ ʃʲː å ɨ̃ᵝ ˈɔ́ː ˌæ͡ɪ ɪ̝ ʼˈə ˈt͡ʂ ˈə̃ ʃ̩̩̩̩̩̩̩̩̩̩ ˌɐ̃ r͈ʲ ʲ̩̩̩ ʼɒ tˣ ˈɔ̌ː a͜oː ŏː χʷ p̩̩ ðː ɡʱː b̩ʲ l͈ ŋ͡mː ʲa ˈô ˌɶː e̝̝ ˈí ˌa̠ ɯ̟̊ᵝ r̝ː ḧ m̂ː ʃ̩̩̩̩̩̩̩̩̩ ˌpʰ ʃ̩̩̩̩̩̩ ˈɛ̝ i̥ k̩̩ t̩ʲ øˑ ɪ̯ː Iᵑ d͡s̻ ˈɭ ˌ˥ ˈj͡ɛ ˈl̥ ˾˧˧ ⁽ʼ e͡ ǧ ˌt͡ɕ ɔⁿ ˈʑ ˌkʰ ˌʋ ˈm̩ ˩˨ ˈɥ y̆ː ʲ̩̩̩̩̩ ˈôː ˈgʲ g̩ t̩̩ ɴː ˈæ͜ɑ a͡ ɯ̟̃ᵝ ʃ̩̩̩̩̩̩̩ ˈpʲ ˈvʲ w̥ ˈˈn ˌˌɫ ɲˀ ʀ̝ ʃːˀ ʃ̩̩̩ ˧˧ʰ Iᵐ æ̃ː ˈˈ ˌ2 ɔʲ t͡sˤ ɔˡ s̠ʰ zˀ lˣ t̪ʼ ɪˠ Ż k̚ʲ ɑ̃ʼ ɨˀ ʒ̩ʲ i̩̩̩ i͡ɛ z̺ ˌħ Üː rʷ ʼk ⁽ ð̩ˠ ˈU bʼ ˈɹ̝ d̥ˀ ʃˠ ə̯ˀ ʁ̝ː ˈd͡s ûˑ ˈɾ̩ː ˌəː ˦˧ p̺ s̺ː y̯ˀ yⁿ æ̹̌ œ̞ˀ ɡˠ ˈɡ̊ i͡ɪ t͡ʂʰ zˤ ˈd͡z ˌA ˌɔ̃ ヮ ő ʃʼ ʱ ˈi͡eː ˈˈp aːː d̪ʰ s̬ ɨ̥ ˈt͡ʃʰ ˌå ẽ̞ː œ͡ɪ ɹ̯ tˡ æ̃ˑ ə͡ɪ ʁʲ ˌæ͡ʊ ヶ k͡pː ɐ́ː ɤ̃ ɫˠ nʷ ʔː ˈæ͜ɑː ˌe͡iː ˌõː ‿ æ͜ə ɒ̯ ˈɾʲ ˛˧ � yᵐ ɐ̯ːˀ ˈẽː d͡ʒʱː i̩ u͡ɛ ʃᵊ ˈĕ ˌő ɐʲ ˈə͡ɪ ˌʐ d̪ʱː ʃ̥ ʼɐ̃ ˈd̪ʱ ˈn̪ ˪˩˧ aˤˀ ɤ̃ː ʀʲ ˈŝ ˌt͡sʰ ˌˈʂ χʲ d͡ʒˠ y͡i ɲ̩ ʲ͡sʲ ˈø͡ɪ ˈˈɡ ˌú ˪ˀ˩ ˫ u̯ˀ æ̌ ɔ̯ː ɾⁿ ᵐb l͡m ð͡ʒ ɲ̠ʲ ʊ̯ː ˈi͡ɪ ˈtʲ 0ᵐ i͜ɑː oʲ yʼ ɐ̯ː ˈI l̻ əʼ ʏ̃ˑ ʼuː ˒ dˮ ɔ̂ ə̝ ɛːː ɡˡ ʁˠ ʌˀ ʌ̹ː ˌm̩ p̪ˠ ɰᵝᵝ ʃ̩̩̩̩̩̩̩̩ ʲˈi ˈn̩ː ˌɶ ːd aˡ ɔ̃ˀ ɪʼ ɾˣ ˈŃ ˈŏ jʱ mʼ oˑ æ̂ˑ ˈˈiː ˌl̩ː ˌɔ̃ː ˒˒ j͡m t̩̩̩̩̩̩̩̩ ʁʼ ˌtʰ ˒ː θˠ -ʲ d̪͡z ʐˤ ˈɒː ˈɽ ˰ k̪ n͈ ə̀ː ɯⁿ ˈǒ ˈu͡ɛ ˌe̞ː ˧͡s dˡ d͡zˠ e̯ː e͡ə i͡m pˢ p̥ xˠ ɑʲ ɕʱ ɤ̂ː ʲˈuʲ ˈɛ͡ʊ ˌɛ̝ ˌʉ f̩̩ kʰˀ ŷ ŋ͡mˠ ɛᵐ ʰ̠ ˈmʲ ˈœ͡ʊ ˌɪː ˾ˠ lʰ ɐ́ˑ ɛˡ ɡʼ ˌɭ Oᵐ nʲʲ s̪ˠ ø̯ˑ ˈæ̃ Yⁿ t͡ʂː ð̞ˀ ɜ͡ ɪ̌ː ɹ̊ ˌi͡ɪ ˌɛ̃ ˌβ ː͡ ˛˧˧ ˟ ˳ 0ⁿ gʼ t̪͡ɕ t͡ç ŭ v̩̩ z͡z ɲᵝ ɹ̝̊ ʼă ʼb ʼˈɜː ʼˌə ˈɫʲ ˈˈe ˌχ ː́ː ˔˧˥ ˧˧˧˧ ˲ ’ d͡s iʲ ũ̯ʼ ɦːˀ ʲˌɛʲ ˈbʱ ˌd͡ʒ ˌɣ ˧˛ kˡ k͡z mʷ s̩̩̩̩ s̪ˑ s͡s æ̂ː ø̞ː ʒʼ ˈõ ˈɛ̂ː ˈɰᵝ ˧˨˩˦ ˹ kːʲ k̩̩̩̩̩̩̩ ɑ̆ ɦʲː ɦᵊ ʊ̃ː ʔ̩ ʼ ʼˈe ˕˥ ˧͡ʒ .ʲ Yᵐ bʷ o̩̩̩ t͡ɕ̚ y͡ə z̩ʲ æ̩ ð͡ʊ ˈlʲ ˈʕ ˌn̩ d͡ʒʼ e͡ɛ kːʲː ņ pʰʲ qː s̝ s̩̩̩ s͡ʑ t͡sʷ uːː w̞ ø͡ʊ ɔ̔ ɣː ʂʼ ˌẽː ˌɔ́ː ˌɛ̌ː θ̞ ほ ヤː k̩̩̩̩̩ l̝̊ s̠ʲ w̝ x̩ ðˠˠ ø̃ ɤ̃ˑ ɹ̝ː ʲi ʼʊ ˈnʲ ˛ː ヮː f̩ʲ lˠː õːˀ r̩̩ s͡ʊ t̻ ǿ ø̥ ɐ̯ˑ ɔ͡ʊ ɗː ʁˡ ˈkʲ ˈʃʰ ˌi̯ ˕˧ ˜˥ ˰˧ c͡p d͡ʒʷ eᵝ j̯ rˡ t͡s̻ː zʼ ø̞ˑ ŋ́ ɑ̂ː əʰ ə̯ˑ ɨʲ ɨ̩ ɫ͡z ɯ̟ᵝᵝ ɾˑ ʃʷ ʃ̩̩̩̩ ʋ̂ ʔʰ ʲˈj͡aː ʲ̩̩ː ˈjː ˈoːː ˈũ ˌĕ ˧˥ʰ ĕː fʰ k̚ʰ t̪͡s̠ û͡ʊ æ̆ ħː ɒ̃ː ɬː ˈbʰ ˌɸ ˩˩ k̚ː r̥ː t̪ˑ üː y̞ ʒ̥ ˈʋʲ ˹˩˧ lᵑ tʷ t͡sˤˤ ə̂ː ɬʼ ʃˑ ˈɡʱ ε b̩̩ ỹː ɐ̃ˑ ɔ̯̃ ɛ̃ʼ ʰ̃ˑ lʼ ỹ œ̃ˑ ɑʼ ʼd͡z ˈǒː ˌɬ ⁿɗⁿ Ǚ t͡ɪ t͡ʃʷ ʊˠ ʔʷ ʼm ˈɰ ˌa̰ ˌɵː lʷ ʼˈæ ˈ- ˈsʲ ˈy͡ɔ ˌá ə̃ˑ ʼp ˈŋ̍ i͡eː r͡s ˈd̪ ˌɑ̃ ˌɛ͡ɪ ˧ˀ ˩˨ˀ˩ dʱ ə̃ː ʁ̩ː madeupword0000 madeupword0001 madeupword0002 madeupword0003 <mask>'
    
    def __init__(self):
        self.symbols = self.SYMBOLS.split()
        self.vocab = {s: i for i, s in enumerate(self.symbols)}
        
        self.pad_token_id = self.vocab.get('<pad>', 1)
        self.unk_token_id = self.vocab.get('<unk>', 3)
        self.sos_token_id = self.vocab.get('<s>', 0)
        self.eos_token_id = self.vocab.get('</s>', 2)
        self.mask_token_id = self.vocab.get('<mask>', len(self.vocab)-1)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        unk = self.unk_token_id
        return [self.vocab.get(c, unk) for c in text.split()]
    
    def __call__(self, text: str, **kwargs) -> dict:
        input_ids = self.encode(text)

        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask) 
        }

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special_tokens = {'<s>', '<pad>', '</s>', '<unk>', '<mask>', 'U'}
        
        if skip_special:
            return "".join(self.id_to_token.get(i, '<unk>') for i in ids 
                           if self.id_to_token.get(i, '<unk>') not in special_tokens)
        
        return "".join(self.id_to_token.get(i, '<unk>') for i in ids)

    def __len__(self):
        return len(self.vocab)


class UromanCharTokenizer:
    def __init__(self):
        base_chars = " abcdefghijklmnopqrstuvwxyz"
        
        self.vocab = {c: i for i, c in enumerate(base_chars)}
        
        self.special_tokens = ["[PAD]", "[UNK]", "[MASK]"]
        
        offset = len(self.vocab)
        for i, t in enumerate(self.special_tokens):
            self.vocab[t] = offset + i
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab["[PAD]"]
        self.unk_token_id = self.vocab["[UNK]"]
        #self.cls_token_id = self.vocab["[CLS]"]
        #self.sep_token_id = self.vocab["[SEP]"]
        self.mask_token_id = self.vocab["[MASK]"]


    def encode(self, text: str) -> List[int]:
        tokens = []
        for c in text:
            if c in self.vocab:
                tokens.append(self.vocab[c])
            else:
                tokens.append(self.unk_token_id)
        return tokens
    
    def __call__(self, text: str, **kwargs) -> dict:
        input_ids = self.encode(text)
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask) 
        }

    
def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "[UNK]")
            if skip_special and token in {"[PAD]", "[UNK]", "[MASK]"}:
                continue
            tokens.append(token)
        return "".join(tokens)