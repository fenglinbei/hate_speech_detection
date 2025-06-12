from typing import List, Dict

def parse_llm_output_quad(llm_output: str) -> List[Dict]:
        """Parse and standardize LLM output"""

        quadruples = []
        group_mapping = {
            'racism': 'Racism',
            'region': 'Region',
            'lgbtq': 'LGBTQ',
            'sexism': 'Sexism',
            'others': 'Others',
            'non_hate': 'non_hate',
            'non-hate': 'non_hate',
            'nonhate': 'non_hate',
            'nohate': 'non_hate',
            'hate': 'hate'
        }
        
        lines = llm_output.strip().split('\n')
        for line in lines:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 4:
                continue

            target = parts[0] if parts[0].upper() != 'NULL' else None
            argument = parts[1] if parts[1].upper() != 'NULL' else None

            tg_raw = parts[2].strip().lower()
            targeted_groups: list[str] = []
            for tg_raw_sp in tg_raw.split(","):
                tg = group_mapping.get(tg_raw_sp.strip(), None)
                if not tg:
                    continue
                targeted_groups.append(tg)
            targeted_group = ", ".join(targeted_groups)
            
            hateful = parts[3].strip().lower()
            hateful = group_mapping.get(hateful, None)
            hateful = hateful if hateful in {'hate', 'non_hate'} else None

            if targeted_group and hateful:
                quadruples.append({
                    'target': target,
                    'argument': argument,
                    'targeted_group': targeted_group,
                    'hateful': hateful
                })
        return quadruples

def parse_llm_output_trip(llm_output: str) -> List[Dict]:
        """Parse and standardize LLM output"""

        quadruples = []
        label_map = {
                "A": "Racism",
                "B": "Region",
                "C": "LGBTQ",
                "D": "Sexism",
                "E": "others",
                "F": "non-hate"
            }
        
        lines = [line.strip() for line in llm_output.strip().split('[SEP]')]
        hateful = "hate"
        for line in lines:
            line = line.split("[END]")[0].strip()
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 3:
                continue

            target = parts[0] if parts[0].upper() != 'NULL' else None
            argument = parts[1] if parts[1].upper() != 'NULL' else None
            label_raw = parts[2].strip()

            targeted_groups: list[str] = []
            for label_raw_sp in label_raw.split(","):
                # label = label_map.get(label_raw_sp.strip(), None)
                label = label_raw_sp.strip()
                if not label:
                    continue
                if label != "non-hate":
                    targeted_groups.append(label)
                else:
                    hateful = "non_hate"
                    targeted_groups.append("non_hate")
                    break

            targeted_group = ", ".join(targeted_groups)

            if targeted_group and hateful:
                quadruples.append({
                    'target': target,
                    'argument': argument,
                    'targeted_group': targeted_group,
                    'hateful': hateful
                })
        return quadruples

def validate_quadruples(quadruples: List[Dict]) -> bool:
        """Validate quadruple format"""

        if not quadruples:
            return False
            
        valid_targeted_groups = {'racism', 'region', 'lgbtq', 'sexism', 'others', 'non_hate'}
        valid_hateful = {'hate', 'non_hate'}
        
        return all(
            (all(s.lower().strip() in valid_targeted_groups for s in q['targeted_group'].split(", "))) and q['targeted_group'] and
            (q['hateful'] in valid_hateful) and q['hateful']
            for q in quadruples
        )

if __name__ == "__main__":
    print(validate_quadruples([
        {
          "target": "基佬",
          "argument": "为什么不怕染艾",
          "targeted_group": "LGBTQ, others",
          "hateful": "hate"
        },
        {
          "target": "基佬",
          "argument": "不要艾就永久单身",
          "targeted_group": "LGBTQ, others",
          "hateful": "hate"
        }
      ]))