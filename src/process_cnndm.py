import sys
import os
import hashlib
import struct
import subprocess
import collections


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '[CLS]'
SENTENCE_END = '[SEP]'

data_dir = "datasets/outdated/cnndm"

all_train_urls = os.path.join(data_dir, "url_lists/all_train.txt")
all_val_urls = os.path.join(data_dir, "url_lists/all_val.txt")
all_test_urls = os.path.join(data_dir, "url_lists/all_test.txt")

cnn_tokenized_stories_dir = os.path.join(data_dir, "cnn_stories_tokenized")
dm_tokenized_stories_dir = os.path.join(data_dir, "dm_stories_tokenized")
finished_files_dir = os.path.join(data_dir, "finished_files_510")

DO_LOWER = False
DO_SEP = False
MAX_TOKENS = 510

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  if DO_LOWER:
    lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  article = ' '.join(article_lines)
  abstract = ' '.join(highlights)

  return article, abstract

def write_to_bin(url_file, out_file_article, out_file_abstract):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  with open(out_file_article, 'w') as article_writer:
    with open(out_file_abstract, 'w') as abstract_writer:
      for idx,s in enumerate(story_fnames):
        if idx % 1000 == 0:
          print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
          story_file = os.path.join(cnn_tokenized_stories_dir, s)
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
          story_file = os.path.join(dm_tokenized_stories_dir, s)
        else:
          print("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
          # Check again if tokenized stories directories contain correct number of files
          print("Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
          check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
          check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
          raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, s))

        # Get the strings to write to file
        article, abstract = get_art_abs(story_file)
        article = ' '.join(article.split(' ')[:MAX_TOKENS])
        abstract = ' '.join(abstract.split(' ')[:MAX_TOKENS])
        eol = "\n"
        if idx == num_stories - 1:
          eol = ""
        article_writer.write(article + eol)
        abstract_writer.write(abstract + eol)

  print("Finished writing file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  # Create some new directories
  if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
  if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.article.txt"), os.path.join(finished_files_dir, "test.abstract.txt"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "valid.article.txt"), os.path.join(finished_files_dir, "valid.abstract.txt"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.article.txt"), os.path.join(finished_files_dir, "train.abstract.txt"))

