import pickle
import re
from nltk import word_tokenize
from sklearn.manifold import TSNE
import pandas as pd
from stop_words import get_stop_words
from plotly.io import write_html
import plotly.express as px
from nltk import download
import dash_dangerously_set_inner_html
from sklearn.metrics.pairwise import cosine_distances
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
# download('punkt')
stop_words = get_stop_words('english')




with open('pickles/tfidf_matrix.pickle', 'rb') as handle:
    tfidf_matrix = pickle.load(handle)

with open('pickles/tfidf_vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

with open('pickles/word2vec.pickle', 'rb') as handle:
    word2vec = pickle.load(handle)

with open('pickles/retrieved_df.pickle', 'rb') as handle:
    retrieved_df = pickle.load(handle)

with open('pickles/sentences_df.pickle', 'rb') as handle:
    sentences_df = pickle.load(handle)

cases_df = pd.read_csv('pickles/cases.csv')

max_cases = cases_df.Cases.max() + 100000
max_deaths = cases_df.Deaths.max() + 100000

fig = px.scatter(cases_df, x="Cases", y="Deaths", text='countriesAndTerritories',
                 animation_frame="dateRep", animation_group="countriesAndTerritories",
                 size='popData2018', color="countriesAndTerritories", hover_name="countriesAndTerritories",
                 size_max=55, log_x=True, log_y=True, range_x=[10, max_cases], range_y=[1, max_deaths])
fig.update_traces(textposition='top center')
fig.update_layout(
    showlegend=False,
    title="The size represents the country's population",
    xaxis_title="Number of Cases",
    yaxis_title="Number of Deaths",
    font=dict(
        size=12
    )
)
fig.update_yaxes(automargin=True)
fig.update_xaxes(automargin=True)


#print(offline.plot(fig, include_plotlyjs=False, output_type='div'))
write_html(fig, 'g.txt', auto_play=False, full_html=False)


def search(query, method='tfidf'):

    # Lower case and remove trailing whitespaces
    query = query.lower().strip()

    if method == 'tfidf':
        query_modified = vectorizer.transform([query])
        sentences_df['Rank'] = cosine_distances(tfidf_matrix, query_modified)
        print(sentences_df[['Id', 'Sentence', 'Rank']].iloc[0:5])
        # display_results(query, sentences_df)
        return sentences_df


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


search_app = [
    #html.H2(children='Literature Search App'),
    html.H4(children='Enter a Search Query'),
    dcc.Input(id='input', placeholder='Example: Incubation period', type='text',
              style={'width': '30%'}),
    html.Button(id='submit-button', type='submit', children='Search'),
    dcc.RadioItems(id='method-radio',
                   options=[
                       {'label': 'TF-IDF', 'value': 'tfidf'},
                       {'label': 'Word Mover\'s Distance', 'value': 'wmd'},
                       {'label': 'BERT', 'value': 'bert'}
                   ],
                   value='tfidf', style={'textAlign': 'center'},
                   labelStyle={'display': 'inline-block', 'padding': '0.5%', 'text-align': 'justify'}),
    html.Div(id='output_div', style={'textAlign': 'left', 'padding': '0%'})
]


word_app = [
    #html.H2(children='Word Embeddings Visualizer'),
    html.H4(children='Enter a Word to Plot Similar Words'),
    dcc.Input(id='word_input', placeholder='Example: Treatment', type='text',
              style={'width': '30%'}),
    html.Button(id='word_submit', type='submit', children='Search')
]


app.layout = html.Div([
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Literature Search App', value='tab-1-example',
                children=[html.Div(children=search_app, style={'textAlign': 'center'})]),
        dcc.Tab(label='Word Embeddings Visualizer', value='tab-2-example',
                children=[html.Div(children=word_app, style={'textAlign': 'center'})]),
        dcc.Tab(
            label='Cases Progression Graph', value='tab-3-example',
            children=[
                html.Div(children=[
                    #html.H2(children='Progression Over Time'),
                    html.H4(children='Click the Play Button below!'),
                    dcc.Graph(id='progression_plot', figure=fig)
                ], style={'textAlign': 'center'}
                )
            ]
        )
    ]),
    html.Div(id='word_output', style={'textAlign': 'center'})
], style={'textAlign': 'center'})


@app.callback(Output('output_div', 'children'),
              [Input('submit-button', 'n_clicks')],
               #Input('method-radio', 'value')],
              [State('input', 'value')])
def update_output(clicks, query):

    if clicks is not None:

        #sentences_df = search(query)
        query = query.lower().strip()
        query_modified = vectorizer.transform([query])
        sentences_df['Rank'] = cosine_distances(tfidf_matrix, query_modified)
        to_display = sentences_df.sort_values('Rank', ascending=True).iloc[0:20]

        query = word_tokenize(query)
        raw_html = ''
        for idx, row in to_display.iterrows():
            title = str(retrieved_df.loc[row.Id].title)
            authors = str(retrieved_df.loc[row.Id].authors).split(',')[0] + '. et al.'
            sentence = word_tokenize(row.Sentence)
            # Display title of the paper and the authors
            raw_html += "<div><b><font color='black'>" + title + '. ' + "</b>" + authors + "</font><br>"

            # Loop over words in the sentence and paste as highlighted text if word is in query,
            # else paste as normal string
            text = str(["<span style='background-color:#9ae59a'>" + word + "</span>"
                        if word in query else word
                        for word in sentence]) + '<hr></div>'
            # Since the text gets pasted like this ['good', 'night'],
            # remove the brackets, commas, and quotes
            text = re.sub("[\'\"\,\[\]]", "", text)

            raw_html += "<p><font color='#3B57BD'>" + text + "</font></p>".strip("\'")

        return dash_dangerously_set_inner_html.DangerouslySetInnerHTML(raw_html)


@app.callback(Output('word_output', 'children'),
              [Input('word_submit', 'n_clicks'), Input('tabs-example', 'value')],
              [State('word_input', 'value')])
def update_word(clicks, tab, word):

    if clicks is not None:
        word = word.lower()
        if len(word.split(' ')) > 1:
            return html.H6(children='Please enter only 1 word, as the embeddings are of unigrams!')

        try:
            words = word2vec.wv.most_similar(word, topn=50)
        except:
            return html.H6(children='The word you entered is not in the vocabulary! Try entering something else.')

        words = [x[0] for x in words] + [word]
        X = word2vec[words]

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.DataFrame(X_tsne, index=words, columns=['x', 'y'])

        text_size = [30 if text == word else 12 for text in df.index]

        fig = {
            'data': [
                {
                    'x': df.x.values,
                    'y': df.y.values,
                    'text': df.index,
                    'name': 'Word Embedding Space',
                    'mode': 'text',
                    'textfont': {'size': text_size}
                }
            ]
        }

        if tab == 'tab-2-example':
            return html.Div(children=[
                dcc.Graph(
                    id='graph-2-tabs',
                    figure=fig,
                    style={'height': '100%'}
                )
            ], style={'textAlign': 'center'})


if __name__ == '__main__':
    app.run_server(debug=True)
