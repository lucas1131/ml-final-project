SELECT
matches.match_id,
matches.radiant_team_id,
matches.dire_team_id,
matches.duration,
matches.radiant_score,
matches.dire_score,
matches.radiant_gold_adv,
matches.radiant_xp_adv,
((team_match.radiant = TRUE) = matches.radiant_win) radiant_win

FROM matches
NATURAL JOIN match_patch
NATURAL JOIN team_match
LEFT JOIN teams on (team_match.team_id = teams.team_id)

ORDER BY matches.match_id DESC nulls LAST, radiant DESC
LIMIT 20000