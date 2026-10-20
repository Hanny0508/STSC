from tts_scores.clvp import CLVPMetric


cv_metric = CLVPMetric(device='cuda')
score = cv_metric.compute_fd('audio/1_to_2_vc.wav', 'audio/2.wav')
print(score)