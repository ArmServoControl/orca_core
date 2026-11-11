from orca_core import load_hand

hand = load_hand(mock=True)      
hand.connect()
hand.init_joints(move_to_neutral=False)

hand.set_joint_positions({"index_mcp": 50})
print(hand.get_joint_position().data["index_mcp"])