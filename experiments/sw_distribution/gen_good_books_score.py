import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

MEDIAN_SW_THRESH = 1
    
def get_good_books(books1, books2, scores):
  score_for_book = {}
  for i in range(len(books1)):
    book1 = int(books1[i])
    book2 = int(books2[i])
    score = scores[i]
    if book1 not in score_for_book:
      score_for_book[book1] = []
    if book2 not in score_for_book:
      score_for_book[book2] = []
    score_for_book[book1].append(score)
    score_for_book[book2].append(score)

  good_books = set()
  for book, book_scores in score_for_book.items():
    if np.median(book_scores) >= MEDIAN_SW_THRESH:
      good_books.add(book)
  return good_books

def gen_good_df(books1, books2, scores, good_books):
  good_books1 = []
  good_books2 = []
  good_scores = []
  for i in range(len(books1)):
    book1 = int(books1[i])
    book2 = int(books2[i])
    score = scores[i]
    if book1 in good_books and book2 in good_books:
      good_books1.append(book1)
      good_books2.append(book2)
      good_scores.append(score)
  good_df = pd.DataFrame({
    'book1':good_books1, 
    'book2':good_books2, 
    'score':good_scores}
  )
  return good_df

if __name__ == "__main__":  
  df = pd.read_csv('data/experiments/sw_distribution/sw_scores_sbert.csv')
  books1 = df['book1'].tolist()
  books2 = df['book2'].tolist()
  scores = df['score'].tolist()
  
  good_books = get_good_books(books1, books2, scores)
  good_df = gen_good_df(books1, books2, scores, good_books)
  good_df.to_csv(
    'data/experiments/sw_distribution/good_sw_scores_sbert.csv', index=False)