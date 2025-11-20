## Example of `gold index change` experiment working documents data object:

* openbook / openbook_random:
```json
{
  "gold_at_4": {
    "questions": [
      "who got the first nobel prize in physics"
    ],
    "answers": [
      [
        "Wilhelm Conrad R\u00f6ntgen"
      ]
    ],
    "documents": [
      [
        {
          "title": "Nobel Prize in Physics",
          "text": "receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was",
          "id": "628725",
          "score": 1.6234328,
          "hasanswer": false,
          "isgold": false,
          "original_retrieval_index": 0
        },
        {
          "title": "Norwegian Americans",
          "text": "science, Ernest Lawrence won the Nobel Prize in Physics in 1939. Lars Onsager won the 1968 Nobel Prize in Chemistry. Norman Borlaug, father of the Green Revolution, won the Nobel Peace Prize in 1970. Christian B. Anfinsen won the Nobel Prize for chemistry in 1972. Ivar Giaever won the Nobel Prize in Physics 1973. Carl Richard Hagen is noted for his work in physics. In engineering, Clayton Jacobson II is credited with the invention of the modern personal watercraft. Ole Singstad was a pioneer of underwater tunnels. Ole Evinrude invented the first outboard motor with practical commercial application, recognizable today",
          "id": "4107064",
          "score": 1.6037977,
          "hasanswer": false,
          "isgold": false,
          "original_retrieval_index": 1
        }
      ]
    ]
  },
  "gold_at_0": {
    "questions": [
      "who got the first nobel prize in physics"
    ],
    "answers": [
      [
        "Wilhelm Conrad R\u00f6ntgen"
      ]
    ],
    "documents": [
      [
        {
          "title": "List of Nobel laureates in Physics",
          "text": "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad R\u00f6ntgen, of Germany, who received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.  John Bardeen is the only laureate to win the prize twice\u2014in 1956 and 1972. Maria Sk\u0142odowska-Curie also won two Nobel Prizes, for physics in 1903 and chemistry in 1911. William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate; he won the prize in 1915 at the age of 25. Two women have won the prize: Curie and Maria Goeppert-Mayer (1963). As of 2017, the prize has been awarded",
          "id": null,
          "score": null,
          "hasanswer": true,
          "isgold": true,
          "original_retrieval_index": null
        },
        {
          "title": "Nobel Prize in Physics",
          "text": "receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was",
          "id": "628725",
          "score": 1.6234328,
          "hasanswer": false,
          "isgold": false,
          "original_retrieval_index": 0
        }
      ]
    ]
  },
  "gold_at_9": {
    "questions": [
      "who got the first nobel prize in physics"
    ],
    "answers": [
      [
        "Wilhelm Conrad R\u00f6ntgen"
      ]
    ],
    "documents": [
      [
        {
          "title": "Nobel Prize in Physics",
          "text": "receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was",
          "id": "628725",
          "score": 1.6234328,
          "hasanswer": false,
          "isgold": false,
          "original_retrieval_index": 0
        },
        {
          "title": "Norwegian Americans",
          "text": "science, Ernest Lawrence won the Nobel Prize in Physics in 1939. Lars Onsager won the 1968 Nobel Prize in Chemistry. Norman Borlaug, father of the Green Revolution, won the Nobel Peace Prize in 1970. Christian B. Anfinsen won the Nobel Prize for chemistry in 1972. Ivar Giaever won the Nobel Prize in Physics 1973. Carl Richard Hagen is noted for his work in physics. In engineering, Clayton Jacobson II is credited with the invention of the modern personal watercraft. Ole Singstad was a pioneer of underwater tunnels. Ole Evinrude invented the first outboard motor with practical commercial application, recognizable today",
          "id": "4107064",
          "score": 1.6037977,
          "hasanswer": false,
          "isgold": false,
          "original_retrieval_index": 1
        }
      ]
    ]
  }
}
```

* baseline:

```json
{
  "baseline": {
    "questions": [
      "who got the first nobel prize in physics"
    ],
    "answers": [
      [
        "Wilhelm Conrad R\u00f6ntgen"
      ]
    ],
    "documents": [
        {
          "title": "List of Nobel laureates in Physics",
          "text": "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad R\u00f6ntgen, of Germany, who received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.  John Bardeen is the only laureate to win the prize twice\u2014in 1956 and 1972. Maria Sk\u0142odowska-Curie also won two Nobel Prizes, for physics in 1903 and chemistry in 1911. William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate; he won the prize in 1915 at the age of 25. Two women have won the prize: Curie and Maria Goeppert-Mayer (1963). As of 2017, the prize has been awarded",
          "id": null,
          "score": null,
          "hasanswer": true,
          "isgold": true,
          "original_retrieval_index": null
        }
      ]
  }
}
```

* closedbook:

```json
{
  "closedbook": {
    "questions": [
      "who got the first nobel prize in physics"
    ],
    "answers": [
      [
        "Wilhelm Conrad R\u00f6ntgen"
      ]
    ]
  }
}
```

## Example of `gold index change` experiment results data object:

* openbook / openbook_random:

```json
{
  "model": "tiiuae/Falcon3-Mamba-7B-Instruct",
  "experiment_type": "gold_index_change",
  "num_documents": 10,
  "prompting_mode": "openbook",
  "execution_date": "2025-04-14 15:22:01",
  "experiments": {
    "gold_at_0": {
      "model_answers": ["The next Deadpool movie is scheduled to be released on December 21, 2019."],
      "scores": [0.0],
      "metric": "best_subspan_em",
      "num_prompt_tokens": [1446]
    },
    "gold_at_4": {
      "model_answers": ["The next Deadpool movie is scheduled to be released on December 21, 2019."],
      "scores": [0.0],
      "metric": "best_subspan_em",
      "num_prompt_tokens": [1446]
    },
    "gold_at_9": {
      "model_answers": ["The next Deadpool movie is scheduled to be released on December 21, 2019."],
      "scores": [0.0],
      "metric": "best_subspan_em",
      "num_prompt_tokens": [1446]
    }
  }
}
```

* closedbook:

```json
{
  "model": "tiiuae/Falcon3-Mamba-7B-Instruct",
  "experiment_type": "gold_idx_change",
  "num_documents": 10,
  "prompting_mode": "openbook",
  "execution_date": "2025-04-14 15:22:01",
  "experiments": {
    "closedbook": {
        "model_answers": ["The next Deadpool movie is scheduled to be released on December 21, 2019."],
        "scores": [0.0],
        "metric": "best_subspan_em",
        "num_prompt_tokens": [1446]
    }
  }
}
```
