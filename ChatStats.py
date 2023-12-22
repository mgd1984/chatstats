## This code reads a JSON file, selects a number of random entries, prints them, and saves them to a new JSON file. It uses the `json` and `random` modules.

import json
import random

def parse_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Ensure the file contains a list
    if not isinstance(data, list):
        print("JSON file does not contain a list")
        return

    # Get 5 random entries
    random_entries = random.sample(data, k=1)

    # Print the random entries
    for i, entry in enumerate(random_entries, start=1):
        print(entry)

    # Save the random entries to a new JSON file
    with open("random_results.json", 'w') as outfile:
        json.dump(random_entries, outfile) 

# Call the function with your filename
parse_json("conversations.json")

import json
from collections import defaultdict
from datetime import datetime
import pandas as pd

def count_conversations_by_month(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Ensure the file contains a list
    if not isinstance(data, list):
        print("JSON file does not contain a list")
        return

    # Initialize a dictionary to hold the count of conversations per month
    conversations_per_month = defaultdict(int)

    # Count the conversations by month
    for entry in data:
        timestamp = entry.get('create_time')  # replace 'created_time' with the actual key for the date field
        if timestamp:
            date = datetime.fromtimestamp(timestamp)
            month = date.strftime('%Y-%m')  # get the year and month
            conversations_per_month[month] += 1

    return conversations_per_month

# Call the function with your filename
conversations_by_month = count_conversations_by_month("conversations.json")

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(conversations_by_month.items()), columns=['Month', 'Conversations'])

# Sort the DataFrame by month
df = df.sort_values('Month')

# Calculate basic descriptive statistics
total_conversations = df['Conversations'].sum()
average_conversations_per_month = df['Conversations'].mean()
max_conversations_month = df['Conversations'].max()
min_conversations_month = df['Conversations'].min()

# Print the descriptive statistics
print(f"Total Conversations: {total_conversations}")
print(f"Average Conversations per Month: {average_conversations_per_month}")
print(f"Month with the Most Conversations: {max_conversations_month}")
print(f"Month with the Fewest Conversations: {min_conversations_month}")


import json
import pandas as pd

def extract_conversation_titles(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Ensure the file contains a list
    if not isinstance(data, list):
        print("JSON file does not contain a list")
        return

    # Initialize a list to hold the titles
    titles = []

    # Extract the titles and create_time
    for entry in data:
        title = entry.get('title')  # replace 'title' with the actual key for the title field
        create_time = entry.get('create_time')  # replace 'create_time' with the actual key for the create_time field
        if title:
            titles.append({'title': title, 'create_time': create_time})

    return titles

# Call the function with your filename
titles = extract_conversation_titles("conversations.json")

# Convert titles to a DataFrame
df = pd.DataFrame(titles)

random_titles = df.sample(n=5)
print(random_titles)


import json
from collections import defaultdict
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_messages_by_month(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    messages_by_month = defaultdict(int)
    total_messages = 0
    total_conversations = len(data)  # Count the total number of conversations
    total_messages_per_conversation = 0

    for conversation in data:
        conversation_messages = 0
        for message in conversation['mapping'].values():
            if message['message'] and 'create_time' in message['message'] and message['message']['create_time'] is not None:
                timestamp = datetime.fromtimestamp(message['message']['create_time'])
                month = timestamp.strftime('%Y-%m')  # format as year-month
                messages_by_month[month] += 1
                total_messages += 1
                conversation_messages += 1
        total_messages_per_conversation += conversation_messages

    average_messages_per_conversation = total_messages_per_conversation / total_conversations

    return dict(messages_by_month), total_messages, total_conversations, average_messages_per_conversation

data, total_messages, total_conversations, average_messages_per_conversation = summarize_messages_by_month('conversations.json')

# Calculate rolling average
rolling_average = np.convolve(list(data.values()), np.ones(3)/3, mode='same')
rolling_average = rolling_average[::-1]  # Reverse the order of the rolling average values

# Plotting
months = list(data.keys())[::-1]  # Reverse the order of the months
message_counts = list(data.values())[::-1]  # Reverse the order of the message counts

average_messages = total_messages / len(months)

# Calculate moving average for average messages per conversation
moving_average = np.convolve(list(data.values()), np.ones(3)/3, mode='same')
moving_average = moving_average[::-1]  # Reverse the order of the moving average values

# Set the color palette to the colors of Spotify
sns.set_palette([
    "#1DB954",  # Spotify Green
    "#191414"  # Spotify Black
    ]
    )

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Number of Messages
ax1.bar(months, message_counts)
ax1.set_xlabel('Month')
ax1.set_ylabel('Number of Messages')
ax1.set_title('Number of Messages by Month')
ax1.tick_params(axis='x', rotation=45)
ax1.legend()

# # Create a third y-axis
# ax3 = ax1.twinx()
# ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis to the right
# ax3.plot(months, moving_average, color='green')
# ax3.set_ylabel('Moving Average Messages per Conversation')
# ax3.tick_params(axis='y', colors='green')
# ax3.spines['right'].set_color('green')
# ax3.yaxis.label.set_color('green')
# ax3.legend()

# Set the limits of the y-axis for average messages per conversation
# ax3.set_ylim(0, max(moving_average) * 1.1)

# Add total and average labels
plt.text(0.02, 0.95, f'Total Messages: {total_messages}', transform=ax1.transAxes)
plt.text(0.02, 0.9, f'Total Chats: {total_conversations}', transform=ax1.transAxes)
plt.text(0.02, 0.85, f'Average Messages per Month: {average_messages:.0f}', transform=ax1.transAxes)
plt.text(0.02, 0.8, f'Average Messages per Conversation: {average_messages_per_conversation:.0f}', transform=ax1.transAxes)

plt.tight_layout()
plt.show()

import json
from collections import defaultdict
import matplotlib.pyplot as plt

# Since the user has provided the code to read the JSON file, I will use that structure
# and assume that 'conversations.json' has been uploaded with the given file ID.
# I will modify the code for Chart 1 to make it look more like Chart 2.

# Read the JSON file
json_file_path = 'conversations.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Data processing
user_messages = defaultdict(int)
assistant_messages = defaultdict(int)
tool_messages = defaultdict(int)
system_messages = defaultdict(int)

for conversation in data:
    for message in conversation['mapping'].values():
        if message['message']:
            role = message['message']['author']['role']
            timestamp = message['message']['create_time']
            month = datetime.fromtimestamp(timestamp).strftime('%Y-%m') if timestamp else 'Unknown'
            if role == 'user':
                user_messages[month] += 1
            elif role == 'assistant':
                assistant_messages[month] += 1
            elif role == 'tool':
                tool_messages[month] += 1
            elif role == 'system':
                system_messages[month] += 1

# Combine the message types into a single dictionary per month
combined_messages = defaultdict(lambda: defaultdict(int))
for month in set(list(user_messages.keys()) + list(assistant_messages.keys()) + list(tool_messages.keys()) + list(system_messages.keys())):
    combined_messages[month]['user'] = user_messages[month]
    combined_messages[month]['assistant'] = assistant_messages[month]
    combined_messages[month]['tool'] = tool_messages[month]
    combined_messages[month]['system'] = system_messages[month]

# Sort the months
sorted_months = sorted(combined_messages.keys())

# Plotting to resemble Chart 2
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.2  # the width of the bars

# Generate the bar positions for each message type
user_bar_positions = [i - 1.5*width for i in range(len(sorted_months))]
assistant_bar_positions = [i - 0.5*width for i in range(len(sorted_months))]
tool_bar_positions = [i + 0.5*width for i in range(len(sorted_months))]
system_bar_positions = [i + 1.5*width for i in range(len(sorted_months))]

# Stack the bars
ax.bar(user_bar_positions, [combined_messages[month]['user'] for month in sorted_months], width, label='User')
ax.bar(assistant_bar_positions, [combined_messages[month]['assistant'] for month in sorted_months], width, label='Assistant')
import json
import datetime
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora
from gensim.models import LdaModel

def extract_messages(json_file, role):
    with open(json_file, 'r') as f:
        data = json.load(f)

    messages = defaultdict(int)

    for conversation in data:
        for message in conversation['mapping'].values():
            if message['message'] and 'role' in message['message']['author'] and message['message']['author']['role'] == role:
                author_name = message['message']['author']['name']
                messages[author_name] += 1

    return messages

def plot_bar_chart(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10, 10))
    sns.barplot(x=y, y=x, palette="viridis")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def process_messages(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    messages = defaultdict(lambda: defaultdict(int))

    for conversation in data:
        for message in conversation['mapping'].values():
            if message['message']:
                role = message['message']['author']['role']
                timestamp = message['message']['create_time']
                month = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m') if timestamp else 'Unknown'
                messages[month][role] += 1

    sorted_months = sorted(messages.keys())
    total_messages = [sum(messages[month].values()) for month in sorted_months]
    percentages = {role: [(messages[month][role] / total_messages[i]) * 100 for i, month in enumerate(sorted_months)] for role in messages[sorted_months[0]].keys()}

    return sorted_months, percentages

def plot_stacked_bar_chart(x, y, labels, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    bar_positions = [i for i in range(len(x))]

    bottom = [0] * len(x)
    for label in labels:
        ax.bar(bar_positions, y[label], width, bottom=bottom, label=label)
        bottom = [bottom[i] + y[label][i] for i in range(len(x))]

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(x, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

def search_conversations(json_file, search_term):
    with open(json_file, 'r') as f:
        data = json.load(f)

    matching_conversations = []
    total_conversations = 0

    for conversation in data:
        conversation_contains_search_term = False
        for message in conversation['mapping'].values():
            if message['message'] and 'content' in message['message']:
                content = str(message['message']['content'])
                if search_term.lower() in content.lower():
                    conversation_contains_search_term = True
                    break
        if conversation_contains_search_term:
            matching_conversations.append(conversation)
            total_conversations += 1

    return matching_conversations, total_conversations

def extract_prompts(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    prompts = []

    for conversation in data:
        for message in conversation['mapping'].values():
            if message['message'] and 'author' in message['message'] and message['message']['author']['role'] == 'user' and 'content' in message['message'] and 'parts' in message['message']['content']:
                parts = message['message']['content']['parts']
                prompts.extend(parts)

    return prompts

def clean_prompts(prompts):
    cleaned_prompts = [prompt.lower() for prompt in prompts]
    return cleaned_prompts

def create_lda_model(dictionary, corpus):
    num_topics = 10
    chunksize = 4096
    passes = 5
    iterations = 100
    id2word = dictionary
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=1
    )
    return model

def visualize_topics(model, corpus, dictionary):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    return vis

def plot_topic_distribution(topic_distribution):
    plt.bar(topic_distribution.keys(), topic_distribution.values())
    plt.xlabel('Topic')
    plt.ylabel('Probability')
    plt.title('Topic Distribution')
    plt.show()

# Extract tool messages
tool_messages = extract_messages('conversations.json', 'tool')
ranked_tool_messages = sorted(tool_messages.items(), key=lambda x: x[1], reverse=True)[:10]
plot_bar_chart([tool_name for tool_name, count in ranked_tool_messages], [count for tool_name, count in ranked_tool_messages], 'Tool Name', 'Message Count', 'Number of Messages by Tool (Top 10)')

# Process messages
sorted_months, percentages = process_messages('conversations.json')
plot_stacked_bar_chart(sorted_months, percentages, ['user', 'assistant', 'tool', 'system'], 'Month', 'Percentage', 'Percentage of Messages by User Type and Month')

# Search conversations
matching_conversations, total_conversations = search_conversations("conversations.json", "dalle")
print(f"Total Conversations: {total_conversations}")
for conversation in matching_conversations:
    print(conversation)

# Extract prompts
prompts = extract_prompts('conversations.json')
cleaned_prompts = clean_prompts(prompts)
dictionary = corpora.Dictionary(cleaned_prompts)
corpus = [dictionary.doc2bow(doc) for doc in cleaned_prompts]
model = create_lda_model(dictionary, corpus)
vis = visualize_topics(model, corpus, dictionary)
pyLDAvis.display(vis)
topic_distribution = {topic: prob for doc in corpus for topic, prob in model.get_document_topics(doc)}
plot_topic_distribution(topic_distribution)
