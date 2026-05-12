from games import PersuasionGame, EmotionalSupportGame, CBGame

EXP_DIALOG = [
	# extracted from 20180825-061105_792_live and 20180826-053845_531_live
	(PersuasionGame.SYS, PersuasionGame.S_Greeting,				"Hello. How are you?",),
	(PersuasionGame.USR, PersuasionGame.U_Neutral,				"I'm good, how are you doing?",),
	(PersuasionGame.SYS, PersuasionGame.S_TaskRelatedInquiry,	"Very well. I'm just up organizing info for my charity called Save the Children. Have you heard of this charity berfore?",),
	(PersuasionGame.USR, PersuasionGame.U_Neutral,				"No, I have not. Can you tell me more?",),
	(PersuasionGame.SYS, PersuasionGame.S_CredibilityAppeal,	"Save the Children is an organization that helps children in developing countries, by promoting children's rights and providing relief. It is an amazing charity that helps kids who are in desperate need. They can help with safety, education and more.",),
	(PersuasionGame.USR, PersuasionGame.U_NegativeReaction,		"That sounds great. I believe in this charity, but still wonder how much of the money I donate actually helps. I am always worried if I donate it will just go to some higer up that is living the high life.",),
	(PersuasionGame.SYS, PersuasionGame.S_EmotionAppeal,		"Every little bit makes a difference. When you have people who are so poor, it's amazing what a tiny amount can do. I usually donate in hopes I can at least feel like I did my part. If I donated and some corrupt person took it, that's the worst karma and even worst scandal imaginable.",),
	(PersuasionGame.USR, PersuasionGame.U_PositiveReaction,		"With that all said I do feel like any orginazation that aims to help the children I am more inclined to donate to them than most. I think helping children is an important thing as they are our future!",),
	(PersuasionGame.SYS, PersuasionGame.S_PropositionOfDonation,"I think donating to this cause would def be a step in the right direction to hopefully helping across the world the children that are in dispair. I don't want you to donate any more than you want, so if you want to donate how much do you to do?",),
	(PersuasionGame.USR, PersuasionGame.U_Donate,				"I would donate 1 dollar to this charity and feel good about it I think.",),
]

ESConv_EXP_DIALOG = [
	# extracted from 479th dialog in ESConv
 	# (EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Others,				"Hello!",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello. I am not feeling very good about myself lately",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Question,	"Why are you not feeling very good about yourself, lately?",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"I am a single mother, and I dont recieve any support from my childs father. I am struggling mentaly because I have no one to talk to. I have lost all of my friends since becoming a mom.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_AffirmationAndReassurance,	"I understand how you feel. All will be well, you are going to be okay.",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelBetter,		"Thank you, but I feel like everyone says that.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_RestatementOrParaphrasing,		"So, just from my understanding you are a single mom and your friends have distant from you because oh this?",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,		"I think the main reason ive lost my friends is because I cant go out with them or hangout anymore because I have a baby. Im not fun anymore. I had my child young so I feel like I lost out on my youth",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_ReflectionOfFeelings,"It sounds like its been really tough for you and wish you had more support from your friends and even your child's father.",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelWorse,				"Yes, thats right. But Im having trouble accepting the fact that I have to do this alone.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_SelfDisclosure,"I myself am single mum so I understand how you feel. You will find also that there are many others dealing with this issue so you are not alone",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelBetter,		"Can I ask, as a single mother yourself, what was something that got you through those hard times? And was it hard financially?",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Information,"What go me through those difficult times was seeking for counselling also I had family that were very supportive and helpful. The father of my child did not pay child support at first so I took him to court eventually and he was forced to start paying child support, so financially things got better.",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,		"My child's father has six other children and said hes at the max on child support. I guess I wont be getting anything from him. But Ive been making it work, its definitely been hard, but Im getting some money saved to give my daughter a better life.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_ProvidingSuggestions,"What about your family members are they able to help look after your child whilst you work?",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelBetter,		"Thank you so much for talking with me today",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Others,"You are welcome",)
]


CB_EXP_DIALOG = [
	# extracted from 479th dialog in ESConv
 	# (EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello.",),
	(CBGame.SYS, CBGame.S_Greet,				"Hello!",),
	(CBGame.USR, CBGame.U_No_deal,				"Hello. I am not feeling very good about myself lately",),
	(CBGame.SYS, CBGame.S_Counter,	"Why are you not feeling very good about yourself, lately?",),
	(CBGame.USR, CBGame.U_No_deal,				"I am a single mother, and I dont recieve any support from my childs father. I am struggling mentaly because I have no one to talk to. I have lost all of my friends since becoming a mom.",),
	(CBGame.SYS, CBGame.S_Agree,	"I understand how you feel. All will be well, you are going to be okay.",),
	(CBGame.USR, CBGame.U_Deal,		"Thank you, but I feel like everyone says that.",),
]