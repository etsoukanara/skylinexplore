# skylinexplore

The code repository for the following paper:

## Skyline-based Termporal Graph Exploration

Evangelia Tsoukanara, Georgia Koloniari, and Evaggelia Pitoura. Skyline-based Termporal Graph Exploration.

Paper accepted at the 27th European Conference on Advances in Databases and Information Systems (ADBIS 2023)

## Abstract
> An important problem in studying temporal graphs is detecting
> interesting events in their evolution, defined as time intervals
> of significant stability, growth, or shrinkage. We consider graphs whose
> nodes have attributes, for example in a network between individuals, the
> attributes may correspond to demographics, such as gender. We build
> aggregated graphs where nodes are grouped based on the values of their
> attributes, and seek for events at the aggregated level, for example, time
> intervals of significant growth between individuals of the same gender. We
> propose a novel approach based on temporal graph skylines. A temporal
> graph skyline considers both the significance of the event (measured by
> the number of graph elements that remain stable, are created, or deleted)
> and the length of the interval when the event appears. We also present
> experimental results of the efficiency and effectiveness of our approach.

## General Information
This repository facilitates the detection of significant events, such as _stability_, _shrinkage_, and _growth_, using temporal graph skylines. The datasets used in this paper is provided in `datasets`.

## Datasets
_DBLP_: directed collaboration dataset that spans over a period of 21 years (2000 to 2020) and includes publicatoins at 21 conferences related to data management research areas. Each node corresponds to an author and is associated with one static (gender), and one time-varying (#publications) attribute.

_MovieLens_: directed mutual rating dataset (built on the benchmark movie ratings dataset) covering a period of six months (May 1st, 2000 to October 31st, 2000) where each node represents a user and an edge denotes that two users have rated the same movie, and is attributed with three static (gender, age, occupation) and one time-varying attribute (average rating per month).

_Primary School_: undirected face-to-face proximity network describing the interactions between students and teachers at a primary school of Lyon, France. The dataset covers a period of 17 hours, each node is associated with two static attributes, gender, and class.

## Dependencies
Python 3.7
