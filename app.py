import streamlit as st
import json
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from streamlit_extras.dataframe_explorer import dataframe_explorer
from datetime import datetime

def parse_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Ensure the file contains a list
    if not isinstance(data, list):
        st.write("JSON file does not contain a list")
        return

    # Get 5 random entries
    random_entries = random.sample(data, k=5)

    # Print the random entries
    for i, entry in enumerate(random_entries, start=1):
        st.write(entry)

    # Save the random entries to a new JSON file
    with open("conversations.json", 'w') as outfile:
        json.dump(random_entries, outfile)

def count_conversations_by_month(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Ensure the file contains a list
    if not isinstance(data, list):
        st.write("JSON file does not contain a list")
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

def extract_conversation_titles(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Ensure the file contains a list
    if not isinstance(data, list):
        st.write("JSON file does not contain a list")
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

def process_messages(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    messages = defaultdict(lambda: defaultdict(int))

    for conversation in data:
        for message in conversation['mapping'].values():
            if message['message']:
                role = message['message']['author']['role']
                timestamp = message['message']['create_time']
                month = datetime.fromtimestamp(timestamp).strftime('%Y-%m') if timestamp else 'Unknown'
                messages[month][role] += 1

    sorted_months = sorted(messages.keys())
    total_messages = [sum(messages[month].values()) for month in sorted_months]
    percentages = {role: [(messages[month][role] / total_messages[i]) * 100 for i, month in enumerate(sorted_months)] for role in messages[sorted_months[0]].keys()}

    return sorted_months, percentages

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

def plot_bar_chart(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10, 10))
    sns.barplot(x=y, y=x, palette="viridis")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    st.pyplot()

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
    st.pyplot()

def plot_topic_distribution(topic_distribution):
    plt.bar(topic_distribution.keys(), topic_distribution.values())
    plt.xlabel('Topic')
    plt.ylabel('Probability')
    plt.title('Topic Distribution')
    st.pyplot()

# Sidebar
st.sidebar.title("ChatStats")
st.file_uploader("Upload JSON file")
st.sidebar.write("Analyze your ChatGPT conversations history")
option = st.sidebar.selectbox("Select an option", ["Parse JSON", "Count Conversations by Month", "Extract Conversation Titles", "Summarize Messages by Month", "Process Messages", "Search Conversations", "Extract Prompts"])

if option == "Parse JSON":
    st.header("Parse JSON")
    filename = st.text_input("Enter the filename")
    if st.button("Parse"):
        parse_json(filename)

elif option == "Count Conversations by Month":
    st.header("Count Conversations by Month")
    filename = st.text_input("Enter the filename")
    if st.button("Count"):
        conversations_by_month = count_conversations_by_month(filename)
        st.write(conversations_by_month)
        # Plot bar chart
        x = list(conversations_by_month.keys())
        y = list(conversations_by_month.values())
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.set_title("Conversations by Month")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Data Explorer
        df = pd.DataFrame(conversations_by_month.items(), columns=['Month', 'Count'])
        with st.expander("Data Explorer"):
            dataframe_explorer(df)

elif option == "Extract Conversation Titles":
    st.header("Extract Conversation Titles")
    filename = st.text_input("Enter the filename")
    if st.button("Extract"):
        titles = extract_conversation_titles(filename)
        st.write(titles)
        # Create dataframe
        df = pd.DataFrame(titles)
        # Plot bar chart
        sns.countplot(data=df, x='title')
        plt.xticks(rotation=45)
        plt.xlabel('Title')
        plt.ylabel('Count')
        plt.title('Conversation Titles')
        plt.show()
        st.pyplot()

elif option == "Summarize Messages by Month":
    st.header("Summarize Messages by Month")
    filename = st.text_input("Enter the filename")
    if st.button("Summarize"):
        messages_by_month, total_messages, total_conversations, average_messages_per_conversation = summarize_messages_by_month(filename)
        st.write(f"Total Messages: {total_messages}")
        st.write(f"Total Conversations: {total_conversations}")
        st.write(f"Average Messages per Conversation: {average_messages_per_conversation}")
        st.write(messages_by_month)
        # Create dataframe
        df = pd.DataFrame(messages_by_month.items(), columns=['Month', 'Count'])
        # Plot bar chart
        sns.barplot(data=df, x='Month', y='Count')
        plt.xticks(rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.title('Messages by Month')
        plt.show()
        st.pyplot()

elif option == "Process Messages":
    st.header("Process Messages")
    filename = st.text_input("Enter the filename")
    if st.button("Process"):
        sorted_months, percentages = process_messages(filename)
        plot_stacked_bar_chart(sorted_months, percentages, ['user', 'assistant', 'tool', 'system'], 'Month', 'Percentage', 'Percentage of Messages by User Type and Month')
        plt.show()
        st.pyplot()

elif option == "Search Conversations":
    st.header("Search Conversations")
    filename = st.text_input("Enter the filename")
    search_term = st.text_input("Enter the search term")
    if st.button("Search"):
        matching_conversations, total_conversations = search_conversations(filename, search_term)
        st.write(f"Total Conversations: {total_conversations}")
        for conversation in matching_conversations:
            st.write(conversation)

elif option == "Extract Prompts":
    st.header("Extract Prompts")
    filename = st.text_input("Enter the filename")
    if st.button("Extract"):
        prompts = extract_prompts(filename)
        cleaned_prompts = clean_prompts(prompts)
        dictionary = corpora.Dictionary(cleaned_prompts)
        corpus = [dictionary.doc2bow(doc) for doc in cleaned_prompts]
        model = create_lda_model(dictionary, corpus)
        vis = visualize_topics(model, corpus, dictionary)
        pyLDAvis.display(vis)
        topic_distribution = {topic: prob for doc in corpus for topic, prob in model.get_document_topics(doc)}
        plot_topic_distribution(topic_distribution)
        plt.show()
        st.pyplot()