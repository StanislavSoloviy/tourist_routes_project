import s from './styles.module.scss';
import { CardList } from "components/card-list";

function SpotsPage() {

    return ( 
        <div className={s.wrapper}>
            <h2>Интересные места</h2>            
            <CardList />            
        </div>
     );
}

export default SpotsPage;