from typing import Sequence


from openai import BaseModel


import math


class Prob(BaseModel):
    token: str
    prob: float


class LogProb(BaseModel):
    token: str
    logprob: float

    @property
    def proba(self) -> float:
        return math.exp(self.logprob)

    def to_prob(self) -> Prob:
        return Prob(token=self.token, prob=self.proba)


class TokenWithLogProbs(BaseModel):
    token: str
    logprob: float  # log probability of the particular token
    top_logprobs: Sequence[LogProb]  # log probability of the top 5 tokens

    def sorted_logprobs(self) -> Sequence[LogProb]:  # Highest to lowest
        return sorted(self.top_logprobs, key=lambda x: x.logprob, reverse=True)

    def sorted_probs(self) -> Sequence[Prob]:
        return [logprob.to_prob() for logprob in self.sorted_logprobs()]


class ResponseWithLogProbs(BaseModel):
    response: str
    content: Sequence[TokenWithLogProbs]  #


## Types to support log probs


class OpenaiResponseWithLogProbs(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        return self.choices[0]["message"]["content"]

    def response_with_logprobs(self) -> ResponseWithLogProbs:
        response = self.first_response
        logprobs = self.choices[0]["logprobs"]["content"]
        parsed_content = [TokenWithLogProbs.model_validate(token) for token in logprobs]
        return ResponseWithLogProbs(response=response, content=parsed_content)

    def first_token_probability_for_target(self, target: str) -> float:
        logprobs = self.response_with_logprobs().content
        first_token = logprobs[0]
        for token in first_token.top_logprobs:
            # print(f"Token: {token.token} Logprob: {token.logprob}")
            if token.token == target:
                token_logprob = token.logprob
                # convert natural log to prob
                return math.exp(token_logprob)
        return 0.0
