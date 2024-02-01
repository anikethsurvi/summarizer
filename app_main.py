#main package
import streamlit as st 
import nltkmodules

#packages
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
import altair as alt 

#lexrank
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# function for lexrank summary
def sumy_summarizer(docx,num=2):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,num)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result 

#summary evaluation
from rouge import Rouge 
def evaluate_summary(summary,reference):
	r = Rouge()
	eval_score = r.get_scores(summary,reference)
	eval_score_df = pd.DataFrame(eval_score[0])
	return eval_score_df

def main():
	page_config = {"page_title":"NLP App","page_icon":"::smiley"}
	st.set_page_config(**page_config)


	st.title("Summarizer App")
	raw_text = st.text_area("Enter Text Here")
	if st.button('Summarize'):
		c1,c2 = st.columns(2)
		with c1:
			with st.expander("original text"):
				st.write(raw_text)

			with c2:
				with st.expander("Lex rank summary"):
					my_summary = sumy_summarizer(raw_text)
					doc_len = {"original":len(raw_text),"summary":len(my_summary)}
					st.write(doc_len)
					st.write(my_summary)

					st.info("Rouge Score")
					eval_df = evaluate_summary(my_summary,raw_text)
					st.dataframe(eval_df.T)
					eval_df['metrics'] = eval_df.index
					c = alt.Chart(eval_df).mark_bar().encode(
						x = 'metrics',y='rouge-1')
					st.altair_chart(c)

        
if __name__ == '__main__':
	main()
