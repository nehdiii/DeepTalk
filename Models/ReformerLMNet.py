from trax import layers as tl
import trax

def ReformerLM(vocab_size=33000, n_layers=2, mode='train', attention_type=tl.SelfAttention):
    

    model = tl.Serial( 
                trax.models.reformer.ReformerLM( 
                vocab_size=vocab_size,
                n_layers=n_layers,
                mode=mode,
                attention_type=attention_type
            )
            , tl.LogSoftmax() 
        )        
   
    return model 