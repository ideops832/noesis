//! Italian text corpus loading and processing.
//!
//! Provides a curated corpus of 300+ Italian sentences across five themes:
//! animals, food, work, weather, and family. Vocabulary is deliberately
//! repeated across sentences (synonyms like gatto/felino/micio, cane/cucciolo)
//! and common verbs (mangia, dorme, corre, gioca, lavora) are shared across
//! themes to enable distributional learning.
//!
//! Two entry points:
//! - [`italian_corpus`]: returns the original 325-sentence static corpus.
//! - [`expanded_corpus`]: generates ~3000 sentences via template combination.

/// Returns a corpus of 300+ Italian sentences covering five themes (backward compatible).
///
/// Themes:
/// - Animals (60+ sentences)
/// - Food (60+ sentences)
/// - Work (60+ sentences)
/// - Weather (60+ sentences)
/// - Family (60+ sentences)
pub fn italian_corpus() -> Vec<&'static str> {
    let mut corpus = Vec::new();
    corpus.extend_from_slice(&ANIMALS);
    corpus.extend_from_slice(&FOOD);
    corpus.extend_from_slice(&WORK);
    corpus.extend_from_slice(&WEATHER);
    corpus.extend_from_slice(&FAMILY);
    corpus
}

/// Generates an expanded corpus of ~3000 sentences using template combination.
///
/// Sentences are built by iterating over arrays of subjects, verbs, adjectives,
/// and locations for each theme, then combining them in natural Italian patterns.
/// The static 325-sentence corpus is included as a subset.
pub fn expanded_corpus() -> Vec<String> {
    let mut corpus: Vec<String> = Vec::with_capacity(3500);

    // Include the original static corpus first.
    for s in italian_corpus() {
        corpus.push(s.to_string());
    }

    // ── Animals ──────────────────────────────────────────────────────────────
    {
        let subjects: &[(&str, &str)] = &[
            ("Il", "gatto"), ("Il", "felino"), ("Il", "micio"),
            ("Il", "cane"), ("Il", "cucciolo"), ("L'", "uccello"),
            ("Il", "pesce"), ("Il", "cavallo"), ("Il", "coniglio"),
            ("La", "tartaruga"), ("Il", "lupo"), ("La", "volpe"),
            ("Il", "delfino"), ("L'", "aquila"), ("Il", "serpente"),
            ("La", "farfalla"), ("La", "gallina"), ("Il", "pappagallo"),
        ];
        let verbs = &[
            "dorme", "mangia", "corre", "gioca", "nuota", "vola",
            "caccia", "salta", "beve", "osserva", "cammina", "si nasconde",
        ];
        let locations = &[
            "", "sul divano", "nel giardino", "nel parco", "nel prato",
            "nella foresta", "nel lago", "sul tavolo", "nella cuccia",
            "nel cortile", "vicino al camino",
        ];
        let adjectives: &[&str] = &[
            "", "grande", "piccolo", "veloce", "lento",
            "nero", "bianco", "giovane", "vecchio",
        ];

        for &(art, subj) in subjects {
            for &verb in verbs {
                for &adj in adjectives {
                    let base = if adj.is_empty() {
                        format!("{}{} {}", art, subj, verb)
                    } else {
                        format!("{}{} {} {}", art, subj, adj, verb)
                    };
                    // Pick 2 locations per (subject, verb, adj) to keep count manageable
                    // Use a deterministic selection based on hash of components
                    let loc_idx = simple_hash(subj, verb, adj) % locations.len();
                    let loc_idx2 = (loc_idx + 3) % locations.len();
                    for &li in &[loc_idx, loc_idx2] {
                        let loc = locations[li];
                        let sentence = if loc.is_empty() {
                            base.clone()
                        } else {
                            format!("{} {}", base, loc)
                        };
                        corpus.push(sentence);
                    }
                }
            }
        }
    }

    // ── Food ─────────────────────────────────────────────────────────────────
    {
        let foods = &[
            "pasta", "pizza", "pane", "riso", "minestra", "carne", "pesce",
            "frutta", "verdura", "torta", "gelato", "formaggio", "insalata",
        ];
        let food_verbs = &[
            "cuoce", "bolle", "profuma", "scalda", "rinfresca",
        ];
        let food_locations = &[
            "nella pentola", "nel forno", "nella cucina", "sul tavolo",
            "nel piatto",
        ];
        let agents = &[
            ("Il", "cuoco"), ("La", "mamma"), ("La", "nonna"),
            ("Il", "papà"), ("Il", "bambino"), ("La", "ragazza"),
            ("Il", "nonno"), ("La", "zia"), ("Lo", "zio"),
        ];
        let agent_actions = &[
            "prepara", "cucina", "mangia", "assaggia", "taglia", "serve",
        ];
        let food_adj = &[
            "", "fresco", "caldo", "buono", "saporito", "dolce",
        ];

        // Pattern 1: "La pasta cuoce nella pentola"
        for &food in foods {
            let art = food_article(food);
            for &verb in food_verbs {
                for &loc in food_locations {
                    corpus.push(format!("{}{} {} {}", art, food, verb, loc));
                }
            }
        }

        // Pattern 2: "Il cuoco prepara la pasta fresca"
        for &(a_art, agent) in agents {
            for &action in agent_actions {
                for &food in foods {
                    let f_art = food_article(food);
                    // Pick an adjective deterministically
                    let adj_idx = simple_hash(agent, action, food) % food_adj.len();
                    let adj = food_adj[adj_idx];
                    let sentence = if adj.is_empty() {
                        format!("{}{} {} {}{}", a_art, agent, action, f_art, food)
                    } else {
                        format!("{}{} {} {}{} {}", a_art, agent, action, f_art, food, adj)
                    };
                    corpus.push(sentence);
                }
            }
        }
    }

    // ── Work ─────────────────────────────────────────────────────────────────
    {
        let workers: &[(&str, &str)] = &[
            ("Il", "programmatore"), ("L'", "ingegnere"), ("Il", "medico"),
            ("L'", "insegnante"), ("L'", "operaio"), ("Il", "contadino"),
            ("Il", "cuoco"), ("Il", "cameriere"), ("Lo", "studente"),
            ("Il", "ricercatore"), ("Il", "muratore"), ("Il", "falegname"),
            ("Il", "meccanico"), ("Il", "giardiniere"), ("Il", "veterinario"),
        ];
        let actions = &[
            "lavora", "scrive", "progetta", "studia", "insegna",
            "costruisce", "ripara", "mangia", "corre", "dorme",
        ];
        let places = &[
            "", "in ufficio", "nella fabbrica", "nel laboratorio",
            "a scuola", "in ospedale", "nel cantiere", "nel negozio",
            "nel parco", "a casa",
        ];
        let time_quals = &[
            "", "ogni giorno", "la mattina", "la sera", "con impegno",
            "con cura", "durante la pausa",
        ];

        for &(art, worker) in workers {
            for &action in actions {
                // Pick 2 place+qualifier combos
                let pi = simple_hash(worker, action, "") % places.len();
                let ti = simple_hash(worker, action, "t") % time_quals.len();
                let pi2 = (pi + 4) % places.len();
                let ti2 = (ti + 3) % time_quals.len();
                for &(p, t) in &[(pi, ti), (pi2, ti2), (pi, ti2)] {
                    let place = places[p];
                    let tq = time_quals[t];
                    let mut s = format!("{}{} {}", art, worker, action);
                    if !place.is_empty() {
                        s.push(' ');
                        s.push_str(place);
                    }
                    if !tq.is_empty() {
                        s.push(' ');
                        s.push_str(tq);
                    }
                    corpus.push(s);
                }
            }
        }
    }

    // ── Weather ──────────────────────────────────────────────────────────────
    {
        let elements: &[(&str, &str)] = &[
            ("Il", "sole"), ("La", "pioggia"), ("Il", "vento"),
            ("La", "neve"), ("La", "nebbia"), ("Il", "temporale"),
            ("Le", "nuvole"), ("La", "grandine"), ("La", "brezza"),
            ("Il", "gelo"), ("Il", "fulmine"), ("Il", "tuono"),
        ];
        let weather_verbs = &[
            "splende", "cade", "soffia", "copre", "arriva",
            "illumina", "rinfresca", "colpisce", "bagna", "porta",
        ];
        let qualifiers = &[
            "", "forte", "leggero", "intenso", "freddo", "caldo",
            "improvviso", "persistente",
        ];
        let weather_locs = &[
            "", "nel cielo", "sulla città", "in montagna", "nella pianura",
            "sulla costa", "nel parco", "sulla campagna", "tra gli alberi",
            "sulle strade",
        ];

        for &(art, elem) in elements {
            for &verb in weather_verbs {
                // 3 combos of qualifier+location
                let qi = simple_hash(elem, verb, "") % qualifiers.len();
                let li = simple_hash(elem, verb, "l") % weather_locs.len();
                let qi2 = (qi + 3) % qualifiers.len();
                let li2 = (li + 4) % weather_locs.len();
                let qi3 = (qi + 5) % qualifiers.len();
                for &(q, l) in &[(qi, li), (qi2, li2), (qi3, li)] {
                    let qual = qualifiers[q];
                    let loc = weather_locs[l];
                    let mut s = format!("{}{}", art, elem);
                    if !qual.is_empty() {
                        s.push(' ');
                        s.push_str(qual);
                    }
                    s.push(' ');
                    s.push_str(verb);
                    if !loc.is_empty() {
                        s.push(' ');
                        s.push_str(loc);
                    }
                    corpus.push(s);
                }
            }
        }
    }

    // ── Family ───────────────────────────────────────────────────────────────
    {
        let members: &[(&str, &str)] = &[
            ("La", "mamma"), ("Il", "papà"), ("La", "nonna"),
            ("Il", "nonno"), ("Il", "fratello"), ("La", "sorella"),
            ("Il", "bambino"), ("La", "bambina"), ("La", "zia"),
            ("Lo", "zio"), ("Il", "figlio"), ("La", "figlia"),
        ];
        let family_actions = &[
            "gioca", "cucina", "legge", "canta", "insegna",
            "accompagna", "abbraccia", "mangia", "corre", "dorme",
            "studia", "lavora",
        ];
        let contexts = &[
            "", "nel giardino", "a casa", "a scuola", "nel parco",
            "in cucina", "in camera", "in salotto", "al tavolo",
        ];
        let with_whom = &[
            "", "con il bambino", "con la famiglia", "con gli amici",
            "con i figli", "con il fratello", "con la sorella",
        ];

        for &(art, member) in members {
            for &action in family_actions {
                // 3 combos of context + companion
                let ci = simple_hash(member, action, "") % contexts.len();
                let wi = simple_hash(member, action, "w") % with_whom.len();
                let ci2 = (ci + 3) % contexts.len();
                let wi2 = (wi + 3) % with_whom.len();
                let ci3 = (ci + 6) % contexts.len();
                for &(c, w) in &[(ci, wi), (ci2, wi2), (ci3, wi)] {
                    let ctx = contexts[c];
                    let wh = with_whom[w];
                    let mut s = format!("{}{} {}", art, member, action);
                    if !ctx.is_empty() {
                        s.push(' ');
                        s.push_str(ctx);
                    }
                    if !wh.is_empty() {
                        s.push(' ');
                        s.push_str(wh);
                    }
                    corpus.push(s);
                }
            }
        }
    }

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    corpus.retain(|s| seen.insert(s.clone()));

    corpus
}

/// Generates parallel sentences where each synonym appears in identical contexts.
///
/// For each synonym group, every template produces one sentence per synonym,
/// ensuring that synonyms like gatto/micio/felino see the exact same surrounding
/// words and therefore converge in distributional similarity.
///
/// Returns ~250+ explicitly parallel sentences.
pub fn synonym_parallel_corpus() -> Vec<String> {
    let mut corpus: Vec<String> = Vec::with_capacity(300);

    // ── Synonym groups ──────────────────────────────────────────────────────

    // Animals
    let animal_synonyms: &[&[&str]] = &[
        &["gatto", "micio", "felino"],
        &["cane", "cucciolo"],
    ];
    let animal_templates = &[
        "Il {} dorme sul divano",
        "Il {} mangia il pesce",
        "Il {} corre nel giardino",
        "Il {} gioca con la palla",
        "Il {} beve il latte",
        "Il {} salta sul tavolo",
        "Il {} dorme nella cuccia",
        "Il {} mangia la carne",
        "Il {} corre nel parco",
        "Il {} osserva gli uccelli",
        "Il {} si nasconde sotto il letto",
        "Il {} caccia nel giardino",
        "Il {} gioca con il bambino",
        "Il {} mangia dalla ciotola",
        "Il {} cammina nel prato",
        "Il {} nuota nel lago",
        "Il {} riposa al sole",
        "Il {} corre veloce",
        "Il {} dorme vicino al camino",
        "Il {} gioca nel cortile",
    ];

    for group in animal_synonyms {
        for template in animal_templates {
            for &synonym in *group {
                corpus.push(template.replace("{}", synonym));
            }
        }
    }

    // Family
    let family_synonyms: &[&[&str]] = &[
        &["mamma", "madre"],
        &["papà", "padre"],
        &["nonna", "nonno"],
    ];
    let family_templates_f = &[
        "La {} cucina nella cucina",
        "La {} legge un libro",
        "La {} abbraccia il bambino",
        "La {} canta una canzone",
        "La {} prepara la cena",
        "La {} gioca con i figli",
        "La {} lavora a casa",
        "La {} insegna con pazienza",
        "La {} corre nel parco",
        "La {} dorme in camera",
        "La {} mangia al tavolo",
        "La {} studia la sera",
        "La {} accompagna il bambino",
        "La {} pulisce la casa",
        "La {} guarda la televisione",
    ];
    let family_templates_m = &[
        "Il {} cucina nella cucina",
        "Il {} legge un libro",
        "Il {} abbraccia il bambino",
        "Il {} canta una canzone",
        "Il {} prepara la cena",
        "Il {} gioca con i figli",
        "Il {} lavora a casa",
        "Il {} insegna con pazienza",
        "Il {} corre nel parco",
        "Il {} dorme in camera",
        "Il {} mangia al tavolo",
        "Il {} studia la sera",
        "Il {} accompagna il bambino",
        "Il {} pulisce la casa",
        "Il {} guarda la televisione",
    ];

    // mamma/madre (feminine)
    for template in family_templates_f {
        for &synonym in &["mamma", "madre"] {
            corpus.push(template.replace("{}", synonym));
        }
    }
    // papà/padre (masculine)
    for template in family_templates_m {
        for &synonym in &["papà", "padre"] {
            corpus.push(template.replace("{}", synonym));
        }
    }
    // nonna/nonno: use gendered templates for each
    let nonna_nonno_templates = &[
        "{} cucina nella cucina",
        "{} legge un libro",
        "{} abbraccia il bambino",
        "{} canta una canzone",
        "{} prepara la cena",
        "{} gioca con i figli",
        "{} lavora a casa",
        "{} insegna con pazienza",
        "{} corre nel parco",
        "{} dorme in camera",
        "{} mangia al tavolo",
        "{} racconta una storia",
        "{} accompagna il bambino",
        "{} guarda la televisione",
        "{} cammina nel giardino",
    ];
    for template in nonna_nonno_templates {
        corpus.push(template.replace("{}", "La nonna"));
        corpus.push(template.replace("{}", "Il nonno"));
    }

    // Cross-gender parallel templates for mamma/papà
    // Use contexts where the article is already a stopword
    let parent_templates = &[
        "{} cucina cena",
        "{} prepara colazione",
        "{} gioca figli giardino",
        "{} legge storia bambino",
        "{} lavora ufficio",
        "{} accompagna figli scuola",
        "{} corre parco mattina",
        "{} dorme divano sera",
        "{} mangia pranzo tavolo",
        "{} canta canzone bambino",
        "{} pulisce casa sabato",
        "{} guarda televisione sera",
        "{} insegna compiti figli",
        "{} abbraccia bambino amore",
        "{} porta figli parco",
    ];
    // Repeat parallel templates 3x to boost their weight in the corpus
    for _ in 0..3 {
        for template in parent_templates {
            for &parent in &["mamma", "papà", "madre", "padre"] {
                corpus.push(template.replace("{}", parent));
            }
        }
    }

    // Food verb synonyms (cuoce/bolle, prepara/cucina)
    let cooking_synonyms: &[&[&str]] = &[
        &["cuoce", "bolle"],
        &["prepara", "cucina"],
    ];
    let cooking_templates_1 = &[
        "La pasta {} nella pentola",
        "Il riso {} nella pentola",
        "La carne {} nel forno",
        "La minestra {} sul fuoco",
        "La verdura {} nella padella",
        "Il pesce {} nella pentola",
        "La patata {} nell'acqua",
        "Il brodo {} sul fuoco",
        "La zuppa {} nella pentola",
        "Il sugo {} a fuoco lento",
    ];
    let cooking_templates_2 = &[
        "La mamma {} la pasta",
        "Il cuoco {} la cena",
        "La nonna {} il dolce",
        "Il papà {} il pranzo",
        "La zia {} la torta",
        "Il nonno {} la minestra",
        "Il bambino {} il panino",
        "La ragazza {} l'insalata",
        "Lo zio {} la carne",
        "Il cameriere {} il piatto",
    ];

    for group in cooking_synonyms {
        let templates = if *group == ["cuoce", "bolle"] {
            cooking_templates_1.as_slice()
        } else {
            cooking_templates_2.as_slice()
        };
        for template in templates {
            for &synonym in *group {
                corpus.push(template.replace("{}", synonym));
            }
        }
    }

    // Weather synonyms
    let weather_synonyms: &[&[&str]] = &[
        &["sole", "astro"],
        &["pioggia", "acqua"],
        &["vento", "brezza"],
    ];
    let weather_templates_sole = &[
        "Il {} splende nel cielo",
        "Il {} illumina la città",
        "Il {} scalda la terra",
        "Il {} sorge la mattina",
        "Il {} tramonta la sera",
        "Il {} brilla forte oggi",
        "Il {} riscalda il giardino",
        "Il {} appare tra le nuvole",
        "Il {} risplende sulla campagna",
        "Il {} colora il tramonto",
    ];
    let weather_templates_pioggia = &[
        "La {} cade forte",
        "La {} bagna le strade",
        "La {} arriva improvvisa",
        "La {} cade sulla città",
        "La {} rinfresca il giardino",
        "La {} colpisce le finestre",
        "La {} cade nel parco",
        "La {} bagna la campagna",
        "La {} arriva dalla montagna",
        "La {} cade leggera",
    ];
    let weather_templates_vento = &[
        "Il {} soffia forte",
        "Il {} muove le foglie",
        "Il {} arriva da nord",
        "Il {} rinfresca la sera",
        "Il {} soffia sulla costa",
        "Il {} porta le nuvole",
        "Il {} muove gli alberi",
        "Il {} soffia nel parco",
        "Il {} arriva improvviso",
        "Il {} rinfresca il giardino",
    ];

    for group in weather_synonyms {
        let templates: &[&str] = match group[0] {
            "sole" => weather_templates_sole,
            "pioggia" => weather_templates_pioggia,
            "vento" => weather_templates_vento,
            _ => continue,
        };
        for template in templates {
            for &synonym in *group {
                corpus.push(template.replace("{}", synonym));
            }
        }
    }

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    corpus.retain(|s| seen.insert(s.clone()));

    corpus
}

/// Generates sentences designed to encode relational parallels for analogy tasks.
///
/// Covers five relational dimensions:
/// 1. Gender/family parallels (mamma:papà, figlia:figlio, nonna:nonno, sorella:fratello)
/// 2. Functional parallels (profession → tool/domain)
/// 3. Temporal/celestial parallels (sole:giorno, luna:notte, estate:inverno)
/// 4. NATURA category boost (nature sentences with shared vocabulary)
/// 5. Action parallels (animal → characteristic action)
///
/// Returns 400+ sentences with parallel structure to support distributional
/// learning of relational analogies.
pub fn relational_corpus() -> Vec<String> {
    let mut corpus: Vec<String> = Vec::with_capacity(500);

    // ── 1. Gender/Family parallels ─────────────────────────────────────────
    // Each template is instantiated for both members of a gendered pair,
    // producing structurally identical contexts that differ only in the
    // gendered word.

    let gender_pairs: &[(&str, &str, &str, &str)] = &[
        // (fem_article, fem_noun, masc_article, masc_noun)
        ("la", "mamma", "il", "papà"),
        ("la", "figlia", "il", "figlio"),
        ("la", "nonna", "il", "nonno"),
        ("la", "sorella", "il", "fratello"),
    ];

    let gender_templates = &[
        "{art} {nome} prepara la cena",
        "{art} {nome} legge un libro",
        "{art} {nome} studia a scuola",
        "{art} {nome} racconta storie",
        "{art} {nome} cammina nel parco",
        "{art} {nome} gioca nel giardino",
        "{art} {nome} mangia a tavola",
        "{art} {nome} dorme in camera",
        "{art} {nome} canta una canzone",
        "{art} {nome} guarda la televisione",
        "{art} {nome} pulisce la casa",
        "{art} {nome} cucina il pranzo",
        "{art} {nome} lavora con impegno",
        "{art} {nome} aiuta in cucina",
        "{art} {nome} va a fare la spesa",
        "{art} {nome} porta il cane a passeggio",
        "{art} {nome} apre la porta",
        "{art} {nome} chiude la finestra",
        "{art} {nome} accende la luce",
        "{art} {nome} spegne il fuoco",
    ];

    for &(f_art, f_noun, m_art, m_noun) in gender_pairs {
        for template in gender_templates {
            corpus.push(
                template
                    .replace("{art}", f_art)
                    .replace("{nome}", f_noun),
            );
            corpus.push(
                template
                    .replace("{art}", m_art)
                    .replace("{nome}", m_noun),
            );
        }
    }

    // Extra cross-pair templates to reinforce within-pair similarity
    let cross_family_templates = &[
        "{art} {nome} abbraccia i bambini",
        "{art} {nome} ride con la famiglia",
        "{art} {nome} parla al telefono",
        "{art} {nome} scrive una lettera",
        "{art} {nome} aspetta alla fermata",
    ];
    for &(f_art, f_noun, m_art, m_noun) in gender_pairs {
        for template in cross_family_templates {
            corpus.push(template.replace("{art}", f_art).replace("{nome}", f_noun));
            corpus.push(template.replace("{art}", m_art).replace("{nome}", m_noun));
        }
    }

    // ── 2. Functional parallels (profession → domain/tool) ────────────────
    // Each template pair maps a profession to its characteristic domain.

    let profession_pairs: &[(&str, &str, &str, &str)] = &[
        // (profession_a, domain_a, profession_b, domain_b)
        ("cuoco", "cucina", "programmatore", "codice"),
        ("dottore", "pazienti", "maestro", "studenti"),
        ("contadino", "terra", "pescatore", "mare"),
        ("pittore", "quadri", "musicista", "musica"),
    ];

    let functional_templates_specific: &[(&str, &str)] = &[
        ("il {prof} lavora in {dom}", "il {prof} lavora con {dom}"),
        ("il {prof} conosce bene {dom}", "il {prof} conosce bene {dom}"),
        ("il {prof} ama {dom}", "il {prof} ama {dom}"),
        ("il {prof} studia {dom}", "il {prof} studia {dom}"),
        ("il {prof} si occupa di {dom}", "il {prof} si occupa di {dom}"),
    ];

    // Shared-context templates (both professions do the same action)
    let functional_shared = &[
        "il {prof} lavora ogni giorno",
        "il {prof} si alza presto la mattina",
        "il {prof} mangia durante la pausa",
        "il {prof} torna a casa la sera",
        "il {prof} guadagna lo stipendio",
        "il {prof} è molto bravo",
        "il {prof} ha molta esperienza",
        "il {prof} inizia a lavorare",
        "il {prof} finisce il lavoro",
        "il {prof} riposa il fine settimana",
    ];

    for &(prof_a, dom_a, prof_b, dom_b) in profession_pairs {
        for &(tmpl_a, tmpl_b) in functional_templates_specific {
            corpus.push(tmpl_a.replace("{prof}", prof_a).replace("{dom}", dom_a));
            corpus.push(tmpl_b.replace("{prof}", prof_b).replace("{dom}", dom_b));
        }
        for tmpl in functional_shared {
            corpus.push(tmpl.replace("{prof}", prof_a));
            corpus.push(tmpl.replace("{prof}", prof_b));
        }
    }

    // Direct profession-domain association sentences
    let profession_domain_direct = &[
        "il cuoco prepara il pranzo in cucina",
        "il cuoco lavora nella cucina del ristorante",
        "il cuoco usa i coltelli in cucina",
        "il programmatore scrive il codice al computer",
        "il programmatore lavora sul codice ogni giorno",
        "il programmatore corregge il codice con attenzione",
        "il dottore cura i pazienti in ospedale",
        "il dottore visita i pazienti ogni giorno",
        "il dottore aiuta i pazienti con cura",
        "il maestro insegna agli studenti a scuola",
        "il maestro guida gli studenti con pazienza",
        "il maestro spiega la lezione agli studenti",
        "il contadino coltiva la terra nei campi",
        "il contadino lavora la terra ogni giorno",
        "il contadino prepara la terra per la semina",
        "il pescatore naviga nel mare aperto",
        "il pescatore lavora nel mare ogni mattina",
        "il pescatore conosce il mare molto bene",
        "il pittore dipinge i quadri nello studio",
        "il pittore crea i quadri con passione",
        "il pittore espone i quadri nella galleria",
        "il musicista suona la musica ogni sera",
        "il musicista compone la musica con talento",
        "il musicista ama la musica da sempre",
    ];
    for &s in profession_domain_direct {
        corpus.push(s.to_string());
    }

    // ── 3. Temporal/celestial parallels ───────────────────────────────────

    // Pairs documented for reference:
    // sole:giorno / luna:notte
    // estate:caldo / inverno:freddo
    // alba:mattina / tramonto:sera

    let temporal_templates_celestial = &[
        "il sole splende di giorno",
        "la luna brilla di notte",
        "il sole illumina il giorno",
        "la luna illumina la notte",
        "di giorno c'è il sole",
        "di notte c'è la luna",
        "il sole scalda di giorno",
        "la luna appare di notte",
        "il sole sorge ogni giorno",
        "la luna sorge ogni notte",
        "il giorno inizia con il sole",
        "la notte inizia con la luna",
        "il sole porta il giorno",
        "la luna porta la notte",
    ];
    for &s in temporal_templates_celestial {
        corpus.push(s.to_string());
    }

    let temporal_templates_seasons = &[
        "estate porta il caldo",
        "inverno porta il freddo",
        "in estate fa caldo",
        "in inverno fa freddo",
        "il caldo arriva in estate",
        "il freddo arriva in inverno",
        "estate significa caldo",
        "inverno significa freddo",
        "durante estate fa molto caldo",
        "durante inverno fa molto freddo",
        "il caldo dell'estate è forte",
        "il freddo dell'inverno è forte",
    ];
    for &s in temporal_templates_seasons {
        corpus.push(s.to_string());
    }

    let temporal_templates_dawn = &[
        "l'alba annuncia la mattina",
        "il tramonto annuncia la sera",
        "la mattina inizia con l'alba",
        "la sera inizia con il tramonto",
        "all'alba comincia la mattina",
        "al tramonto comincia la sera",
        "l'alba colora la mattina",
        "il tramonto colora la sera",
        "ogni mattina arriva l'alba",
        "ogni sera arriva il tramonto",
    ];
    for &s in temporal_templates_dawn {
        corpus.push(s.to_string());
    }

    // ── 4. NATURA category boost ──────────────────────────────────────────
    // Dense nature sentences with heavily shared vocabulary to tighten the
    // distributional cluster for nature words.

    let natura_sentences = &[
        "il sole scalda la terra",
        "la pioggia bagna i fiori",
        "il vento muove le foglie",
        "la neve copre le montagne",
        "il mare è calmo oggi",
        "il fiume scorre nella valle",
        "il lago riflette le montagne",
        "il bosco è pieno di alberi",
        "la foresta è verde e fitta",
        "il cielo è azzurro e limpido",
        "le stelle brillano nel cielo",
        "la luna illumina il lago",
        "il sole tramonta sul mare",
        "la pioggia cade sulle foglie",
        "il vento soffia tra gli alberi",
        "la neve cade sulle montagne",
        "il mare bagna la spiaggia",
        "il fiume attraversa il bosco",
        "il lago è circondato da montagne",
        "il bosco ospita molti animali",
        "la terra è bagnata dalla pioggia",
        "i fiori crescono nel prato",
        "le foglie cadono in autunno",
        "le montagne toccano il cielo",
        "il mare è profondo e blu",
        "il sole riscalda il prato",
        "la pioggia rinfresca la terra",
        "il vento porta le nuvole",
        "la neve copre il prato",
        "il mare riflette il sole",
        "il cielo si copre di nuvole",
        "le stelle illuminano la notte",
        "la luna sorge dietro le montagne",
        "il sole sorge dal mare",
        "la pioggia bagna la terra",
        "il vento muove le onde del mare",
        "la neve copre gli alberi",
        "il mare si calma la sera",
        "il fiume scorre verso il mare",
        "il lago è limpido come il cielo",
        "il bosco profuma di resina",
        "la foresta è silenziosa",
        "i fiori profumano il giardino",
        "le foglie verdi coprono gli alberi",
        "le montagne sono alte e maestose",
        "il sole splende sulla campagna",
        "la pioggia nutre la terra",
        "il vento accarezza i fiori",
        "la neve bianca copre tutto",
        "il mare ondeggia dolcemente",
        "il cielo rosso annuncia il tramonto",
        "il sole caldo scalda i fiori",
        "la pioggia leggera bagna il prato",
        "il vento fresco soffia dal mare",
        "la neve fresca copre la terra",
        "il mare blu brilla al sole",
        "il fiume limpido scorre tra le rocce",
        "il lago calmo riflette il cielo",
        "il bosco verde è pieno di vita",
        "la terra fertile produce frutti",
    ];
    for &s in natura_sentences {
        corpus.push(s.to_string());
    }

    // ── 5. Action parallels (animal → characteristic action) ──────────────

    // Pairs documented for reference:
    // uccello:vola / pesce:nuota
    // gatto:corre / cavallo:galoppa
    // serpente:striscia / rana:salta

    // uccello/pesce specific
    let bird_fish = &[
        "l'uccello vola nel cielo",
        "il pesce nuota nel mare",
        "l'uccello vola tra le nuvole",
        "il pesce nuota tra le alghe",
        "l'uccello vola alto",
        "il pesce nuota in profondità",
        "l'uccello vola ogni giorno",
        "il pesce nuota ogni giorno",
        "l'uccello vola verso il sole",
        "il pesce nuota verso il fondo",
        "l'uccello mangia i semi",
        "il pesce mangia le alghe",
        "l'uccello vive nel nido",
        "il pesce vive nel mare",
        "l'uccello canta la mattina",
        "il pesce si muove in silenzio",
    ];
    for &s in bird_fish {
        corpus.push(s.to_string());
    }

    // gatto/cavallo specific
    let cat_horse = &[
        "il gatto corre nel prato",
        "il cavallo corre nel campo",
        "il gatto corre veloce",
        "il cavallo galoppa veloce",
        "il gatto corre e salta",
        "il cavallo galoppa e salta",
        "il gatto corre nel giardino",
        "il cavallo galoppa nella campagna",
        "il gatto corre dietro al topo",
        "il cavallo galoppa nella prateria",
        "il gatto si muove con agilità",
        "il cavallo si muove con forza",
        "il gatto vive in casa",
        "il cavallo vive nella stalla",
        "il gatto mangia la carne",
        "il cavallo mangia il fieno",
    ];
    for &s in cat_horse {
        corpus.push(s.to_string());
    }

    // serpente/rana specific
    let snake_frog = &[
        "il serpente striscia nell'erba",
        "la rana salta nello stagno",
        "il serpente striscia silenzioso",
        "la rana salta con forza",
        "il serpente striscia sul terreno",
        "la rana salta sulla foglia",
        "il serpente striscia lentamente",
        "la rana salta velocemente",
        "il serpente si nasconde tra le rocce",
        "la rana si nasconde tra le piante",
        "il serpente vive nel prato",
        "la rana vive nello stagno",
        "il serpente caccia i topi",
        "la rana caccia gli insetti",
        "il serpente si muove senza zampe",
        "la rana si muove saltando",
    ];
    for &s in snake_frog {
        corpus.push(s.to_string());
    }

    // Shared animal templates for action parallels
    let shared_animal_templates = &[
        "il {anim} cerca il cibo",
        "il {anim} si riposa al sole",
        "il {anim} è un animale",
        "il {anim} vive nella natura",
        "il {anim} cresce in fretta",
    ];
    let all_animals = &["uccello", "pesce", "gatto", "cavallo", "serpente", "rana"];
    for tmpl in shared_animal_templates {
        for &animal in all_animals {
            let s = tmpl.replace("{anim}", animal);
            // Fix article for uccello
            let s = if animal == "uccello" {
                s.replace("il uccello", "l'uccello")
            } else {
                s
            };
            corpus.push(s);
        }
    }

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    corpus.retain(|s| seen.insert(s.clone()));

    corpus
}

/// Simple deterministic hash for template slot selection (no randomness needed).
fn simple_hash(a: &str, b: &str, c: &str) -> usize {
    let mut h: usize = 5381;
    for byte in a.bytes().chain(b.bytes()).chain(c.bytes()) {
        h = h.wrapping_mul(33).wrapping_add(byte as usize);
    }
    h
}

/// Returns the correct Italian article + space for a food word.
fn food_article(food: &str) -> &'static str {
    match food.chars().next() {
        Some('a') | Some('e') | Some('i') | Some('o') | Some('u') => "l'",
        _ => match food {
            "zucchero" | "zucchina" => "lo ",
            "pasta" | "pizza" | "patata" | "panna" | "minestra"
            | "carne" | "frutta" | "verdura" | "torta" | "insalata" => "la ",
            _ => "il ",
        }
    }
}

// ── Animals (65 sentences) ──────────────────────────────────────────────────

const ANIMALS: [&str; 65] = [
    "Il gatto dorme sul divano",
    "Il felino dorme sul divano morbido",
    "Il micio mangia il pesce fresco",
    "Il gatto mangia la carne",
    "Il felino corre nel giardino",
    "Il micio gioca con la palla",
    "Il gatto corre veloce nel prato",
    "Il cane abbaia forte nel cortile",
    "Il cucciolo gioca con il bambino",
    "Il cane mangia la carne nella ciotola",
    "Il cucciolo dorme nella cuccia calda",
    "Il cane corre nel parco grande",
    "Il gatto si lava le zampe con cura",
    "Il felino caccia il topo nel campo",
    "Il micio dorme vicino al camino",
    "Il gatto beve il latte dalla ciotola",
    "Il cucciolo impara a camminare piano",
    "Il cane nuota nel lago profondo",
    "Il felino osserva gli uccelli dal balcone",
    "Il gatto salta sul tavolo della cucina",
    "Il micio miagola per avere il cibo",
    "Il cane protegge la casa di notte",
    "Il cucciolo mangia il biscotto con appetito",
    "Il gatto insegue la farfalla nel giardino",
    "Il felino si arrampica sull'albero alto",
    "Il micio gioca con il gomitolo di lana",
    "Il cane cammina al guinzaglio con il padrone",
    "Il cucciolo corre dietro alla palla rossa",
    "Il gatto dorme sulla sedia della nonna",
    "Il felino mangia il pesce ogni giorno",
    "Il micio beve l'acqua dalla fontana",
    "Il cane gioca con gli altri cani al parco",
    "Il cucciolo dorme tra le coperte calde",
    "Il gatto graffia il mobile del salotto",
    "L'uccello canta la mattina presto",
    "Il pesce nuota nel fiume limpido",
    "La tartaruga cammina lenta nel giardino",
    "Il coniglio mangia la carota fresca",
    "Il cavallo corre nella campagna verde",
    "La mucca mangia l'erba nel prato",
    "Il gallo canta all'alba ogni mattina",
    "La pecora dorme nel recinto sicuro",
    "Il maiale mangia nella porcilaia sporca",
    "L'aquila vola alta nel cielo azzurro",
    "Il delfino nuota veloce nel mare aperto",
    "La volpe caccia nel bosco fitto",
    "Il lupo corre nella foresta oscura",
    "Il gatto e il cane giocano insieme",
    "Il felino e il cucciolo dormono vicini",
    "Il micio e il cane mangiano dalla stessa ciotola",
    "Il gatto osserva il pesce nell'acquario",
    "Il cane porta il bastone al padrone",
    "Il cucciolo impara i comandi del padrone",
    "Il gatto fa le fusa sul divano",
    "Il felino caccia i topi nella cantina",
    "Il micio dorme tutto il giorno pigro",
    "Il cane aspetta il padrone alla porta",
    "Il cucciolo gioca con il giocattolo nuovo",
    "Il gatto si nasconde sotto il letto",
    "Il felino salta da un mobile all'altro",
    "La gallina depone le uova nel nido",
    "Il pappagallo parla e ripete le parole",
    "Il criceto corre nella ruota veloce",
    "Il serpente striscia nell'erba alta",
    "La farfalla vola tra i fiori colorati",
];

// ── Food (65 sentences) ─────────────────────────────────────────────────────

const FOOD: [&str; 65] = [
    "La pasta cuoce nella pentola grande",
    "La pizza esce dal forno caldo",
    "Il pane fresco profuma nella cucina",
    "Il riso bolle nell'acqua salata",
    "La minestra scalda il corpo d'inverno",
    "Il formaggio stagiona nella cantina buia",
    "Il prosciutto viene tagliato a fette sottili",
    "L'insalata fresca rinfresca d'estate",
    "Il pomodoro matura nell'orto soleggiato",
    "La frutta fresca sta nella ciotola grande",
    "Il bambino mangia la pasta con il sugo",
    "La mamma cucina la cena per la famiglia",
    "Il cuoco prepara il piatto del giorno",
    "La nonna cucina la torta di mele",
    "Il papà griglia la carne nel giardino",
    "La famiglia mangia insieme la domenica",
    "Il ragazzo beve il succo di arancia",
    "La ragazza mangia la frutta dopo pranzo",
    "Il nonno beve il vino rosso a tavola",
    "La zia prepara i biscotti per merenda",
    "Il gelato si scioglie al sole estivo",
    "La torta esce dal forno profumata",
    "Il caffè caldo scalda la mattina fredda",
    "Il latte fresco viene dalla fattoria",
    "Il miele dolce viene dalle api laboriose",
    "Il pane caldo esce dal forno ogni mattina",
    "La pasta al pomodoro piace a tutti",
    "La pizza margherita è la più popolare",
    "Il risotto ai funghi cuoce piano",
    "La lasagna cuoce nel forno per ore",
    "Lo zucchero dolce si scioglie nel caffè",
    "Il burro si scioglie nella padella calda",
    "L'olio d'oliva condisce l'insalata verde",
    "Il peperoncino piccante brucia la bocca",
    "La cipolla fa piangere quando si taglia",
    "L'aglio profuma il sugo della pasta",
    "Il basilico fresco decora la pizza",
    "Il prezzemolo verde condisce il pesce",
    "La carota dolce cresce nell'orto",
    "La patata bolle nell'acqua per il purè",
    "Il cuoco mangia il piatto che ha preparato",
    "La mamma assaggia la minestra calda",
    "Il bambino mangia il gelato con gioia",
    "La ragazza cucina la pasta per gli amici",
    "Il ragazzo prepara il panino per pranzo",
    "La nonna mangia la frutta ogni mattina",
    "Il nonno cucina il pesce alla griglia",
    "Il papà prepara la colazione per tutti",
    "La zia mangia la torta con il caffè",
    "Lo zio beve la birra fredda d'estate",
    "La cena è pronta sulla tavola grande",
    "Il pranzo della domenica unisce la famiglia",
    "La colazione è il pasto più importante",
    "La merenda pomeridiana dà energia",
    "Il cibo italiano è famoso nel mondo",
    "La dieta mediterranea è molto salutare",
    "Il mercato vende frutta e verdura fresca",
    "Il supermercato ha tutto per la cucina",
    "Il ristorante serve piatti tradizionali",
    "La trattoria cucina ricette della nonna",
    "Il forno sforna pane e dolci ogni giorno",
    "La pasticceria prepara torte e biscotti",
    "Il panificio vende pane caldo ogni mattina",
    "Il bar serve caffè e cornetti freschi",
    "La gelateria prepara gelato artigianale",
];

// ── Work (65 sentences) ──────────────────────────────────────────────────────

const WORK: [&str; 65] = [
    "Il lavoratore arriva in ufficio presto",
    "La collega lavora al computer tutto il giorno",
    "Il direttore gestisce l'azienda con cura",
    "La segretaria risponde al telefono ogni mattina",
    "Il programmatore scrive codice al computer",
    "L'ingegnere progetta il ponte nuovo",
    "Il medico visita i pazienti in ospedale",
    "L'insegnante spiega la lezione agli studenti",
    "L'avvocato difende il cliente in tribunale",
    "Il commerciante vende prodotti al mercato",
    "L'operaio lavora nella fabbrica grande",
    "Il contadino lavora nei campi ogni giorno",
    "Il muratore costruisce la casa nuova",
    "L'elettricista ripara l'impianto elettrico",
    "L'idraulico aggiusta il tubo rotto",
    "Il falegname costruisce mobili in legno",
    "Il meccanico ripara la macchina rotta",
    "Il cuoco lavora nel ristorante del centro",
    "Il cameriere serve i clienti al tavolo",
    "Il barista prepara il caffè ogni mattina",
    "Lo studente studia per l'esame difficile",
    "La professoressa corregge i compiti degli studenti",
    "Il ricercatore lavora in laboratorio tutto il giorno",
    "La scienziata studia le molecole al microscopio",
    "Il giornalista scrive l'articolo per il giornale",
    "Il fotografo scatta le foto per il giornale",
    "L'artista dipinge il quadro nello studio",
    "Il musicista suona il piano ogni sera",
    "L'architetto disegna il progetto della casa",
    "Il pilota guida l'aereo verso la destinazione",
    "L'autista guida l'autobus per la città",
    "Il postino consegna le lettere ogni mattina",
    "Il pompiere spegne l'incendio nella foresta",
    "Il poliziotto controlla il traffico in centro",
    "Il soldato difende il paese con coraggio",
    "Il marinaio naviga nel mare aperto",
    "Il pescatore pesca nel fiume ogni mattina",
    "Il giardiniere cura le piante nel parco",
    "Il veterinario cura gli animali malati",
    "Il farmacista vende le medicine in farmacia",
    "L'infermiere assiste i pazienti in ospedale",
    "Il dentista cura i denti dei pazienti",
    "Il panettiere sforna il pane ogni mattina",
    "Il macellaio taglia la carne nel negozio",
    "Il fruttivendolo vende frutta fresca al mercato",
    "Il sarto cuce i vestiti su misura",
    "Il barbiere taglia i capelli nel salone",
    "Il lavoratore mangia il pranzo in mensa",
    "La collega beve il caffè durante la pausa",
    "Il direttore lavora fino a sera tardi",
    "Il programmatore corre nel parco dopo il lavoro",
    "L'ingegnere gioca a calcio con i colleghi",
    "Il medico dorme poco durante il turno",
    "L'insegnante mangia il pranzo a scuola",
    "Lo studente lavora part-time nel negozio",
    "Il ricercatore pubblica i risultati dello studio",
    "La riunione inizia alle nove di mattina",
    "Il progetto deve essere consegnato entro venerdì",
    "Il contratto viene firmato dal direttore",
    "La presentazione spiega il nuovo prodotto",
    "Il colloquio di lavoro dura trenta minuti",
    "Lo stipendio viene pagato ogni fine mese",
    "Le ferie estive durano due settimane",
    "La formazione dei nuovi dipendenti inizia lunedì",
    "Il team lavora insieme al progetto importante",
];

// ── Weather (65 sentences) ───────────────────────────────────────────────────

const WEATHER: [&str; 65] = [
    "Il sole splende nel cielo azzurro",
    "La pioggia cade forte sulle strade",
    "Il vento soffia forte tra gli alberi",
    "La neve copre le montagne d'inverno",
    "Il temporale arriva dalla montagna",
    "L'arcobaleno appare dopo la pioggia",
    "La nebbia copre la pianura al mattino",
    "Il gelo ricopre i vetri delle finestre",
    "La grandine danneggia le colture nei campi",
    "Il tuono rimbomba nella valle profonda",
    "Il fulmine illumina il cielo di notte",
    "La brezza fresca rinfresca la sera estiva",
    "Il caldo afoso opprime la città d'estate",
    "Il freddo pungente arriva con l'inverno",
    "La primavera porta fiori e colori nuovi",
    "L'autunno colora le foglie di rosso e giallo",
    "L'estate porta il sole e il caldo intenso",
    "L'inverno porta la neve e il freddo forte",
    "La temperatura sale durante il giorno",
    "La temperatura scende durante la notte",
    "Il cielo sereno promette una bella giornata",
    "Le nuvole grigie annunciano la pioggia",
    "Il cielo coperto non lascia passare il sole",
    "La foschia limita la visibilità sulla strada",
    "La brina copre l'erba ogni mattina fredda",
    "Il ghiaccio rende le strade pericolose",
    "La siccità colpisce le campagne in estate",
    "L'alluvione allaga le strade della città",
    "Il tornado distrugge le case nella pianura",
    "La tromba d'aria solleva gli oggetti in alto",
    "Il bambino gioca sotto la pioggia leggera",
    "Il cane corre nella neve fresca e soffice",
    "Il gatto dorme vicino al camino caldo",
    "La famiglia resta in casa durante il temporale",
    "Il contadino lavora sotto il sole cocente",
    "Il pescatore aspetta che il vento si calmi",
    "Il marinaio controlla il cielo prima di partire",
    "Lo sciatore scende dalla montagna innevata",
    "Il turista gode il sole sulla spiaggia",
    "Il bambino costruisce il pupazzo di neve",
    "La pioggia bagna i fiori del giardino",
    "Il vento porta le foglie lungo la strada",
    "La neve si scioglie con il sole di primavera",
    "Il temporale dura pochi minuti ma è intenso",
    "La nebbia si alza lentamente verso mezzogiorno",
    "Il sole tramonta dietro le montagne rosse",
    "La luna splende nel cielo stellato di notte",
    "Le stelle brillano nel cielo limpido",
    "L'alba colora il cielo di rosa e arancio",
    "Il tramonto tinge il mare di colori caldi",
    "La temperatura mite rende piacevole la passeggiata",
    "Il clima mediterraneo è caldo e soleggiato",
    "La stagione delle piogge porta acqua ai campi",
    "Il vento freddo del nord porta la neve",
    "La brezza marina rinfresca la costa d'estate",
    "Il cielo azzurro e limpido invita a uscire",
    "Le nuvole bianche decorano il cielo azzurro",
    "La pioggia leggera rinfresca l'aria calda",
    "Il vento forte piega gli alberi nel parco",
    "La neve fresca scricchiola sotto i piedi",
    "Il ghiaccio si forma sulle pozzanghere gelate",
    "La rugiada brilla sull'erba al mattino presto",
    "Il sole caldo scalda la pelle d'estate",
    "La pioggia battente riempie i fiumi e i laghi",
    "Il vento caldo del sud porta la sabbia",
];

// ── Family (65 sentences) ────────────────────────────────────────────────────

const FAMILY: [&str; 65] = [
    "La mamma abbraccia il bambino con amore",
    "Il papà gioca con i figli nel giardino",
    "La nonna racconta le storie della sera",
    "Il nonno insegna a pescare al nipote",
    "La sorella studia nella sua stanza tranquilla",
    "Il fratello gioca a calcio nel cortile",
    "La zia prepara la torta per la festa",
    "Lo zio porta i regali per i nipoti",
    "Il bambino corre nel parco con gli amici",
    "La bambina gioca con le bambole in camera",
    "La famiglia cena insieme ogni sera",
    "Il padre lavora per mantenere la famiglia",
    "La madre cucina il pranzo della domenica",
    "I genitori accompagnano i figli a scuola",
    "I nonni visitano i nipoti ogni settimana",
    "I fratelli giocano insieme nel giardino",
    "Le sorelle condividono la stanza da letto",
    "Lo zio e la zia arrivano per le vacanze",
    "I cugini giocano insieme durante le feste",
    "La famiglia festeggia il compleanno del bambino",
    "La mamma legge la storia della buonanotte",
    "Il papà insegna a guidare la bicicletta",
    "La nonna cucina i biscotti per i nipoti",
    "Il nonno cammina nel parco ogni mattina",
    "La sorella aiuta il fratello con i compiti",
    "Il fratello protegge la sorella più piccola",
    "La zia lavora come insegnante a scuola",
    "Lo zio guida il trattore nella campagna",
    "Il bambino dorme nel lettino con il peluche",
    "La bambina mangia la merenda dopo la scuola",
    "La famiglia va in vacanza al mare d'estate",
    "Il padre porta la famiglia in montagna",
    "La madre prepara le valigie per il viaggio",
    "I genitori leggono ai figli prima di dormire",
    "I nonni raccontano storie del passato",
    "I fratelli litigano ma poi fanno pace",
    "Le sorelle cucinano insieme la torta",
    "Lo zio racconta barzellette divertenti",
    "La zia regala vestiti nuovi ai nipoti",
    "I cugini corrono nel prato verde insieme",
    "La mamma canta la ninna nanna al bambino",
    "Il papà ripara il giocattolo rotto del figlio",
    "La nonna lavora a maglia la sera davanti alla televisione",
    "Il nonno legge il giornale ogni mattina presto",
    "La sorella dipinge i quadri nella sua stanza",
    "Il fratello suona la chitarra la sera",
    "La famiglia mangia la pizza il sabato sera",
    "Il bambino gioca con il gatto nel giardino",
    "La bambina corre con il cane nel parco",
    "La mamma lavora in ufficio tutto il giorno",
    "Il papà cucina la cena quando torna a casa",
    "La nonna dorme il pomeriggio dopo pranzo",
    "Il nonno gioca a carte con gli amici",
    "La sorella mangia la frutta dopo pranzo",
    "Il fratello beve il latte ogni mattina",
    "La famiglia guarda il film insieme la sera",
    "I genitori lavorano e i nonni curano i nipoti",
    "Il bambino impara a leggere con la mamma",
    "La bambina impara a scrivere con il papà",
    "La famiglia celebra le feste tutti insieme",
    "Il padre insegna i valori importanti ai figli",
    "La madre ascolta i problemi dei figli",
    "I nonni sono la memoria della famiglia",
    "I fratelli crescono e diventano grandi insieme",
    "La famiglia è il posto più sicuro del mondo",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_size() {
        let corpus = italian_corpus();
        assert!(
            corpus.len() >= 300,
            "Corpus should have at least 300 sentences, got {}",
            corpus.len()
        );
    }

    #[test]
    fn test_corpus_themes() {
        assert!(ANIMALS.len() >= 60, "Animals should have 60+ sentences");
        assert!(FOOD.len() >= 60, "Food should have 60+ sentences");
        assert!(WORK.len() >= 60, "Work should have 60+ sentences");
        assert!(WEATHER.len() >= 60, "Weather should have 60+ sentences");
        assert!(FAMILY.len() >= 60, "Family should have 60+ sentences");
    }

    #[test]
    fn test_corpus_no_empty() {
        let corpus = italian_corpus();
        for (i, sentence) in corpus.iter().enumerate() {
            assert!(
                !sentence.trim().is_empty(),
                "Sentence {i} should not be empty"
            );
        }
    }

    #[test]
    fn test_expanded_corpus_size() {
        let corpus = expanded_corpus();
        println!("Expanded corpus size: {}", corpus.len());
        assert!(
            corpus.len() >= 2500,
            "Expanded corpus should have at least 2500 sentences, got {}",
            corpus.len()
        );
        // Verify no empty sentences
        for (i, sentence) in corpus.iter().enumerate() {
            assert!(
                !sentence.trim().is_empty(),
                "Expanded sentence {i} should not be empty"
            );
        }
        // Verify it includes the static corpus
        let static_corpus = italian_corpus();
        for s in &static_corpus {
            assert!(
                corpus.iter().any(|c| c == s),
                "Expanded corpus should include static sentence: {}",
                s
            );
        }
    }

    #[test]
    fn test_corpus_has_shared_vocabulary() {
        let corpus = italian_corpus();
        let text = corpus.join(" ").to_lowercase();
        // Common verbs should appear across themes
        assert!(text.contains("mangia"), "Corpus should contain 'mangia'");
        assert!(text.contains("dorme"), "Corpus should contain 'dorme'");
        assert!(text.contains("corre"), "Corpus should contain 'corre'");
        assert!(text.contains("gioca"), "Corpus should contain 'gioca'");
        assert!(text.contains("lavora"), "Corpus should contain 'lavora'");
        // Synonyms
        assert!(text.contains("gatto"), "Corpus should contain 'gatto'");
        assert!(text.contains("felino"), "Corpus should contain 'felino'");
        assert!(text.contains("micio"), "Corpus should contain 'micio'");
        assert!(text.contains("cane"), "Corpus should contain 'cane'");
        assert!(text.contains("cucciolo"), "Corpus should contain 'cucciolo'");
    }
}
