import re
from .symbols import symbols
from .TTScleaners import english_cleaners



_symbol_to_id = {s:i for i,s in enumerate(symbols)}
_id_to_symbol = {i:s for i,s in enumerate(symbols)}


_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def text_to_sequence(text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1)))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  # Append EOS token
  sequence.append(_symbol_to_id['~'])
  return sequence

def string_chunks(ch,chunk_size):
  l = ch.split()
  out = []
  for i in range(int(len(l)/chunk_size)+1):
    out.append(" ".join(l[i*chunk_size:(i+1)*chunk_size]))
  return out


def _clean_text(text):
  return english_cleaners(text)

def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s != '_' and s != '~'