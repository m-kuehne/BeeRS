# BeeRS (B̲e̲st Inte̲rvention for R̲ecommender S̲ystems)

This repository contains the code accompanying our publication.

📚 Full documentation is available at:  
👉 [https://beers.readthedocs.io/en/latest/](https://beers.readthedocs.io/en/latest/)

## Citing BeeRS

If you use BeeRS in your work please cite [our paper](https://doi.org/10.48550/arXiv.2502.12973).

```
@InProceedings{kuehne2025,
  title = 	 {Optimizing Social Network Interventions via Hypergradient-Based Recommender System Design},
  author =       {K\"{u}hne, Marino and Grontas, Panagiotis D. and De Pasquale, Giulia and Belgioioso, Giuseppe and Dorfler, Florian and Lygeros, John},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {31860--31875},
  year = 	 {2025},
  editor = 	 {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v267/main/assets/kuhne25a/kuhne25a.pdf},
  url = 	 {https://proceedings.mlr.press/v267/kuhne25a.html},
  abstract = 	 {Although social networks have expanded the range of ideas and information accessible to users, they are also criticized for amplifying the polarization of user opinions. Given the inherent complexity of these phenomena, existing approaches to counteract these effects typically rely on handcrafted algorithms and heuristics. We propose an elegant solution: we act on the network weights that model user interactions on social networks (e.g., ranking of users’ shared content in feeds), to optimize a performance metric (e.g., minimize polarization), while users’ opinions follow the classical Friedkin-Johnsen model. Our formulation gives rise to a challenging, large-scale optimization problem with non-convex constraints, for which we develop a gradient-based algorithm. Our scheme is simple, scalable, and versatile, as it can readily integrate different, potentially non-convex, objectives. We demonstrate its merit by: (i) rapidly solving complex social network intervention problems with 4.8 million variables based on the Reddit, LiveJournal, and DBLP datasets; (ii) outperforming competing approaches in terms of both computation time and disagreement reduction.}
}
```
