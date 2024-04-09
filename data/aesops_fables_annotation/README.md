Download and unzip annotated_data.zip. 

It contains 153 csv files. 

Let's go through one example file named _eagleFox_11339__eagleFox_34588.csv_ 

The csv file contains alignment for the Aesop's fable _THE EAGLE AND THE FOX_ from two different project Gutenberg books with id 11339 and 34588. 

The books can be found in these links: 
https://www.gutenberg.org/ebooks/11339
https://www.gutenberg.org/ebooks/34588

The columns in the csv files along with their description are as follows: 

sent1	manual_alignments	manual_alignments_1_1	sent2	chat_gpt_alignments	alignment_tool_alignments

| Column     |  Description    |
|-------------|-------------|
| sent1 | sentences from the 1st book's story |
| manual_alignments | alignment pairs (many-to-many alignment) |
| manual_alignments_1_1 | alignment pairs (1-to-1 alignment) |
| sent2 | sentences from the 2nd book's story |
| chat_gpt_alignments | pairs from ChatGPT3.5 |
| alignment_tool_alignments | pairs from GNAT |
