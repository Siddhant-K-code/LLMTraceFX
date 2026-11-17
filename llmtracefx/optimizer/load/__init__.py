"""Concurrent load generation for LLMTraceFX.

``dispatcher`` is a transport-free scheduler that issues requests under a
closed-loop concurrency, an open-loop request rate, or a list of arrival
timestamps, and reports how faithfully it kept to that schedule.
``api_worker`` binds it to the OpenAI-compatible stream collector. The
sweep and goodput layers build on both.
"""
